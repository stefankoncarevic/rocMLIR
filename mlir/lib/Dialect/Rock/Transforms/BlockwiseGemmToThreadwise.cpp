//===- BlockwiseGemmToThreadwise - MLIR Rock ops lowering passes ---===//
//
// Copyright 2020 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================
//
// This pass converts rock.blockwise_* ops to rock.threadwise_*
// and lowers other higher-level ops like transform and fill in preparation for
// the threadwise lowering
//
//===-----------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Tuning/GeneralGemmBlockStructure.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Dialect/Rock/IR/AccelEmitter.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKBLOCKWISEGEMMTOTHREADWISEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-blockwise-to-threadwise"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::rock;
using namespace mlir::affine;

namespace {
struct RockLowerBlockwiseGemmToThreadwisePass
    : public rock::impl::RockBlockwiseGemmToThreadwisePassBase<
          RockLowerBlockwiseGemmToThreadwisePass> {
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// Fill lowering.
//===----------------------------------------------------------------------===//

struct FillRewritePattern : public OpConversionPattern<FillOp> {
  using OpConversionPattern<FillOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(FillOp op, FillOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();
    MemRefType inputType = op.getInput().getType();
    ArrayRef<int64_t> inputShape = inputType.getShape();
    llvm::SmallVector<int64_t> lbs(inputShape.size(), 0);
    llvm::SmallVector<int64_t> strides(inputShape.size(), 1);

    affine::buildAffineLoopNest(
        b, loc, lbs, inputShape, strides,
        [value = adaptor.getValue(), input = adaptor.getInput()](
            OpBuilder &b, Location loc, ValueRange ivs) {
          b.create<memref::StoreOp>(loc, value, input, ivs);
        });

    b.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// BlockwiseFill lowering.
//===----------------------------------------------------------------------===//

struct BlockwiseFillRewritePattern
    : public OpConversionPattern<BlockwiseFillOp> {
  using OpConversionPattern<BlockwiseFillOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BlockwiseFillOp op, BlockwiseFillOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MemRefType memrefType = op.getMemref().getType();
    ArrayRef<int64_t> memrefShape = memrefType.getShape();
    BottomUpTMBuilder threadsToMemrefTrBuilder(rewriter, memrefShape, loc);
    SmallVector<StringRef, 1> lowerNameRefs;
    threadsToMemrefTrBuilder.getStartNames(lowerNameRefs);
    int64_t blockSize = op.getBlockSize();

    Value val = op.getValue();
    int64_t numElements = memrefType.getNumElements();
    Type valueType = val.getType();
    int64_t valueItems = 1;
    Type valueElementType = valueType;
    if (VectorType valueVecType = dyn_cast<VectorType>(val.getType())) {
      valueItems = valueVecType.getNumElements();
      valueElementType = valueVecType.getElementType();
    }
    // guranteed by op verifier that vector length is a factor of memref size
    int64_t numValues = numElements / valueItems;
    int64_t iterLen = ((numValues + blockSize - 1) / blockSize) * valueItems;

    threadsToMemrefTrBuilder.pad(lowerNameRefs[0],
                                 {0, blockSize * iterLen - numElements});
    TransformMapAttr pad = threadsToMemrefTrBuilder.get();

    threadsToMemrefTrBuilder =
        BottomUpTMBuilder::above(threadsToMemrefTrBuilder, pad);
    threadsToMemrefTrBuilder.unmerge({"tid", "iter"}, {0, 1}, lowerNameRefs[0],
                                     {blockSize, iterLen});
    TransformMapAttr unmerge = threadsToMemrefTrBuilder.get();

    gpu::AddressSpaceAttr privateMemoryAddressSpace =
        rewriter.getAttr<gpu::AddressSpaceAttr>(
            gpu::GPUDialect::getPrivateAddressSpace());
    MemRefType valueRegType = MemRefType::get(
        valueItems, valueElementType, AffineMap{}, privateMemoryAddressSpace);
    GpuAllocOp valueReg = rewriter.create<GpuAllocOp>(loc, valueRegType);
    Value zero = rewriter.createOrFold<ConstantIndexOp>(loc, 0);
    rewriter.create<InBoundsStoreOp>(loc, val, valueReg, zero);
    Value tid =
        rewriter.createOrFold<rock::WorkitemIdOp>(loc, rewriter.getIndexType());
    rewriter.create<ThreadwiseWriteAllOp>(
        loc, valueReg, op.getMemref(), rewriter.getArrayAttr({unmerge, pad}),
        /*extraIndices=*/ValueRange{tid}, GemmFeatures::none, StoreMethod::Set,
        true, true);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// BlockwiseGemm lowering.
//===----------------------------------------------------------------------===//

// The structure of this lowing is documented at
// https://github.com/ROCm/rocMLIR/issues/719
struct BlockwiseGemmRewritePattern
    : public OpConversionPattern<BlockwiseGemmOp> {
  using OpConversionPattern<BlockwiseGemmOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(BlockwiseGemmOp op,
                                BlockwiseGemmOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();

    // Prepare some useful constants.
    Value zeroConstantOp = b.createOrFold<ConstantIndexOp>(loc, 0);

    MemRefType blockAType = op.getMatrixA().getType(),
               blockBType = op.getMatrixB().getType(),
               bufferCType = op.getMatrixC().getType();

    auto elementType = bufferCType.getElementType();

    int64_t k = blockAType.getShape()[0];
    int64_t m = blockAType.getShape()[1];
    int64_t n = blockBType.getShape()[1];
    int64_t kPack = blockAType.getShape()[2];

    // Non-accelerator path.

    // Obtain critical attributes.
    int64_t mC = bufferCType.getShape()[0];
    int64_t nC = bufferCType.getShape()[1];

    GeneralGemmParamsAttr params = op.getParams();
    uint32_t blockSize = params.getBlockSize();
    int64_t kPerThread = params.getKPerThread();
    int64_t mPerThread = params.getMPerThread();
    int64_t nPerThread = params.getNPerThread();

    GeneralGemmBlockStructure blockStructure =
        *deriveGeneralGemmBlockStructure(blockSize);

    int64_t mThreadsPerCuwave = blockStructure.mThreadsPerCuwave;
    int64_t nThreadsPerCuwave = blockStructure.nThreadsPerCuwave;
    int64_t cuwaveLen = mThreadsPerCuwave * nThreadsPerCuwave;

    int64_t mCuwavesPerBlock = blockStructure.mCuwavesPerBlock;
    int64_t nCuwavesPerBlock = blockStructure.nCuwavesPerBlock;
    int64_t numCuwaves = mCuwavesPerBlock * nCuwavesPerBlock;
    int64_t derivedBlockSize = numCuwaves * cuwaveLen;
    assert(blockSize == derivedBlockSize &&
           "block structure parameters must multiply to block size");

    int64_t mRepeat = mC / mPerThread;
    int64_t nRepeat = nC / nPerThread;

    if (mRepeat * mCuwavesPerBlock * mThreadsPerCuwave * mPerThread != m)
      return op.emitOpError("The m turing attributes don't multiply to M_LDS");
    if (nRepeat * nCuwavesPerBlock * nThreadsPerCuwave * nPerThread != n)
      return op.emitOpError("The n turing parameters don't multiply to N_LDS");

    LLVM_DEBUG(llvm::dbgs()
               << "M: " << m << "\n"
               << "mRepeat: " << mRepeat << "\n"
               << "mCuwavesPerBlock: " << mCuwavesPerBlock << "\n"
               << "mThreadsPerCuwave: " << mThreadsPerCuwave << "\n"
               << "mPerThread: " << mPerThread << "\n"
               << "n: " << n << "\n"
               << "nRepeat: " << nRepeat << "\n"
               << "nCuwavesPerBlock: " << nCuwavesPerBlock << "\n"
               << "nThreadsPerCuwave: " << nThreadsPerCuwave << "\n"
               << "nPerThread: " << nPerThread << "\n");

    auto ldsTidSplitter = [&](StringRef repeatName, int64_t repeatLen,
                              StringRef perThreadName,
                              int64_t perThreadLen) -> TopDownTMBuilder {
      TopDownTMBuilder splitTidForLDS(
          b, {"k", repeatName, "tid", perThreadName, "kpack"},
          {k, repeatLen, blockSize, perThreadLen, kPack}, loc);
      splitTidForLDS.passThrough({"k", repeatName});
      splitTidForLDS.merge({"m_cuwaves", "n_cuwaves", "m_cuwave", "n_cuwave"},
                           {2, 3, 4, 5}, "tid",
                           {mCuwavesPerBlock, nCuwavesPerBlock,
                            mThreadsPerCuwave, nThreadsPerCuwave});
      splitTidForLDS.passThrough({perThreadName, "kpack"}, {6, 7},
                                 {perThreadName, "kpack"});
      return splitTidForLDS;
    };

    int64_t copyMPerThread = op.getInMPerThread();
    int64_t copyNPerThread = op.getInNPerThread();

    TopDownTMBuilder splitTidA =
        ldsTidSplitter("m_repeat", mRepeat, "m_thread", mPerThread);
    TransformMapAttr splitTidAAttr = splitTidA.get();
    auto toLdsIndexA = TopDownTMBuilder::below(splitTidA, splitTidAAttr);
    toLdsIndexA.passThrough("k");
    toLdsIndexA.unmerge(
        "m", 1, {"m_repeat", "m_cuwaves", "m_cuwave", "m_thread"},
        {mRepeat, mCuwavesPerBlock, mThreadsPerCuwave, mPerThread});
    toLdsIndexA.ignore("n_cuwaves");
    toLdsIndexA.ignore("n_cuwave");
    toLdsIndexA.passThrough({"kpack"}, {2}, {"kpack"});
    TransformMapAttr toLdsIndexAAttr = toLdsIndexA.get();
    SmallVector<Attribute> transformAttrsA{splitTidAAttr, toLdsIndexAAttr};

    // If the dimension `m` has been rotated to minimize bank conflicts we want
    // to apply the same rotation reading from LDS. This rotation happens in
    // `wrapLDSforStore` from
    // mlir/lib/Dialect/Rock/Transforms/GridwiseGemmToBlockwise.cpp which needs
    // to be kept in sync with this function
    int64_t strideA = (kPack == 1 ? copyMPerThread : 1);
    rotateIf(op.getRotateMWithK(), toLdsIndexA, toLdsIndexAAttr, strideA, "m",
             m, 1, "k", k, {"k"}, {"kpack"}, transformAttrsA);

    TopDownTMBuilder splitTidB =
        ldsTidSplitter("n_repeat", nRepeat, "n_thread", nPerThread);
    TransformMapAttr splitTidBAttr = splitTidB.get();
    auto toLdsIndexB = TopDownTMBuilder::below(splitTidB, splitTidBAttr);
    toLdsIndexB.passThrough("k");
    toLdsIndexB.unmerge(
        "n", 1, {"n_repeat", "n_cuwaves", "n_cuwave", "n_thread"},
        {nRepeat, nCuwavesPerBlock, nThreadsPerCuwave, nPerThread});
    toLdsIndexB.ignore("m_cuwaves");
    toLdsIndexB.ignore("m_cuwave");
    toLdsIndexB.passThrough({"kpack"}, {2}, {"kpack"});
    TransformMapAttr toLdsIndexBAttr = toLdsIndexB.get();
    SmallVector<Attribute> transformAttrsB{splitTidBAttr, toLdsIndexBAttr};

    // If the dimension `d` has been rotated to minimize bank conflicts we want
    // to apply the same rotation reading from LDS. This rotation happens in
    // `wrapLDSforStore` from
    // mlir/lib/Dialect/Rock/Transforms/GridwiseGemmToBlockwise.cpp which needs
    // to be kept in sync with this function
    int64_t strideB = (kPack == 1 ? copyNPerThread : 1);
    rotateIf(op.getRotateNWithK(), toLdsIndexB, toLdsIndexBAttr, strideB, "n",
             n, 1, "k", k, {"k"}, {"kpack"}, transformAttrsB);

    Value matrixA, matrixB;
    ArrayAttr transformsA, transformsB;
    bool ldsANeedsi64, ldsBNeedsi64;
    std::tie(matrixA, transformsA, ldsANeedsi64) =
        untransform(b, adaptor.getMatrixA(), b.getArrayAttr(transformAttrsA));
    std::tie(matrixB, transformsB, ldsBNeedsi64) =
        untransform(b, adaptor.getMatrixB(), b.getArrayAttr(transformAttrsB));
    if (ldsANeedsi64 || ldsBNeedsi64)
      return b.notifyMatchFailure(loc, "LDS map can't need 64-bit indexing");

    int64_t threadANumRegisters = kPerThread * mC * kPack;
    int64_t threadBNumRegisters = kPerThread * nC * kPack;

    // Alloc register for thread_a and thread_b.
    auto privateMemoryAddressSpace = b.getAttr<gpu::AddressSpaceAttr>(
        gpu::GPUDialect::getPrivateAddressSpace());
    auto threadARegisterMemRefType =
        MemRefType::get(threadANumRegisters, elementType, AffineMap{},
                        privateMemoryAddressSpace);
    auto threadAAllocOp = b.create<GpuAllocOp>(loc, threadARegisterMemRefType);

    auto threadBRegisterMemRefType =
        MemRefType::get(threadBNumRegisters, elementType, AffineMap{},
                        privateMemoryAddressSpace);
    auto threadBAllocOp = b.create<GpuAllocOp>(loc, threadBRegisterMemRefType);

    // Define views of register tiles for copies
    BottomUpTMBuilder viewA(b, {"raw"}, {threadANumRegisters}, loc);
    viewA.unmerge({"k", "m_repeat", "tid", "m_thread", "kpack"},
                  {0, 1, 2, 3, 4}, "raw",
                  {kPerThread, mRepeat, 1, mPerThread, kPack});
    TransformMapAttr threadACopyViewAttr = viewA.get();

    BottomUpTMBuilder viewB(b, {"raw"}, {threadBNumRegisters}, loc);
    viewB.unmerge({"k", "n_repeat", "tid", "n_thread", "kpack"},
                  {0, 1, 2, 3, 4}, "raw",
                  {kPerThread, nRepeat, 1, nPerThread, kPack});
    TransformMapAttr threadBCopyViewAttr = viewB.get();

    // Main loop.
    Value workitem = b.createOrFold<rock::WorkitemIdOp>(loc, b.getIndexType());
    LLVM_DEBUG(llvm::dbgs() << "Outer loop:\n "
                            << "k =  " << k << "\n"
                            << " kPerThread = " << kPerThread << "\n");
    auto loopOp =
        b.replaceOpWithNewOp<affine::AffineForOp>(op, 0, k, kPerThread);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(loopOp.getBody());
    Value kOffset = loopOp.getInductionVar();

    SmallVector<Value, 5> registerStartCoords(5, zeroConstantOp);
    SmallVector<Value, 5> ldsBufferAStartCoords = {
        kOffset, zeroConstantOp, workitem, zeroConstantOp, zeroConstantOp};
    auto copyALoop = b.create<TransformingForOp>(
        loc, ArrayRef<ValueRange>{ldsBufferAStartCoords, registerStartCoords},
        ArrayRef<Attribute>{transformsA, b.getArrayAttr(threadACopyViewAttr)},
        ArrayRef<int64_t>{kPerThread, mRepeat, 1, mPerThread, kPack},
        /*strides=*/std::nullopt, /*forceUnroll=*/true, /*indexDiffs=*/true);
    {
      OpBuilder::InsertionGuard copyAGuard(b);
      b.setInsertionPointToStart(copyALoop.getBody());
      Value aCopy = b.create<memref::LoadOp>(
          loc, matrixA, copyALoop.getLowerCoords(/*domain=*/0));
      Value aCast = createTypeConversionOp(b, loc, aCopy, elementType);
      b.create<memref::StoreOp>(loc, aCast, threadAAllocOp,
                                copyALoop.getLowerCoords(/*domain=*/1));
    }

    SmallVector<Value, 5> ldsBufferBStartCoords = {
        kOffset, zeroConstantOp, workitem, zeroConstantOp, zeroConstantOp};
    auto copyBLoop = b.create<TransformingForOp>(
        loc, ArrayRef<ValueRange>{ldsBufferBStartCoords, registerStartCoords},
        ArrayRef<Attribute>{transformsB, b.getArrayAttr(threadBCopyViewAttr)},
        ArrayRef<int64_t>{kPerThread, nRepeat, 1, nPerThread, kPack},
        /*strides=*/std::nullopt, /*forceUnroll=*/true, /*indexDiffs=*/true);
    {
      OpBuilder::InsertionGuard copyBGuard(b);
      b.setInsertionPointToStart(copyBLoop.getBody());
      Value bCopy = b.create<memref::LoadOp>(
          loc, matrixB, copyBLoop.getLowerCoords(/*domain=*/0));
      Value bCast = createTypeConversionOp(b, loc, bCopy, elementType);
      b.create<memref::StoreOp>(loc, bCast, threadBAllocOp,
                                copyBLoop.getLowerCoords(/*domain=*/1));
    }

    Value reshapedARegisters = reshapeBuffer(
        b, loc, threadAAllocOp, {"k", "m", "kpack"}, {kPerThread, mC, kPack});
    Value reshapedBRegisters = reshapeBuffer(
        b, loc, threadBAllocOp, {"k", "n", "kpack"}, {kPerThread, nC, kPack});
    // Actually do the gemm - this goes inside the look over kOffset
    b.create<ThreadwiseGemmOp>(loc, reshapedARegisters, reshapedBRegisters,
                               op.getMatrixC());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// BlockwiseGemmAccel lowering.
//===----------------------------------------------------------------------===//
struct BlockwiseGemmAccelRewritePattern
    : public OpConversionPattern<BlockwiseGemmAccelOp> {
  using OpConversionPattern<BlockwiseGemmAccelOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(BlockwiseGemmAccelOp op,
                                BlockwiseGemmAccelOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();

    StringAttr arch = op.getArchAttr();
    RockAccelTuningParamAttrInterface tuningParams = op.getParams();
    int64_t kpackPerBlock = tuningParams.getKpackPerBlock();
    int64_t mPerWave = tuningParams.getMPerWave();
    int64_t nPerWave = tuningParams.getNPerWave();

    Type bufferElemTypeA =
        cast<MemRefType>(adaptor.getMatrixA().getType()).getElementType();
    Type bufferElemTypeB =
        cast<MemRefType>(adaptor.getMatrixB().getType()).getElementType();
    Type dataTypeA = bufferElemTypeA, dataTypeB = bufferElemTypeB;
    if (auto bufferVecTypeA = dyn_cast<VectorType>(bufferElemTypeA))
      dataTypeA = bufferVecTypeA.getElementType();
    if (auto bufferVecTypeB = dyn_cast<VectorType>(bufferElemTypeB))
      dataTypeB = bufferVecTypeB.getElementType();

    auto accelEmitterPtr = rock::accel::AccelEmitter::select(
        op.getFeatures(), dataTypeA, dataTypeB, arch, tuningParams);

    if (!accelEmitterPtr)
      return op.emitOpError("Unable to emit accelerator code.");

    // Extract relevant accelerator parameters
    rock::accel::AccelEmitterParams params = accelEmitterPtr->getParams();
    Type argTypeA = params.argTypeA;
    Type argTypeB = params.argTypeB;
    int64_t mRepeats = params.mRepeats;
    int64_t nRepeats = params.nRepeats;
    int64_t kBase = params.kBase;
    int64_t kBasePerThread = params.kBasePerThread;

    auto tid = b.create<WorkitemIdOp>(loc, b.getIndexType());

    LLVM_DEBUG(llvm::dbgs()
               << "argVectorType A: " << argTypeA << "\n"
               << "argVectorType B: " << argTypeB << "\n"
               << "kBase: " << kBase << "\n"
               << "mPerWave: " << mPerWave << "\n"
               << "nPerWave: " << nPerWave << "\n"
               << "mRepeat: " << mRepeats << "\n"
               << "nRepeat: " << nRepeats << "\n"
               << "kBasePerThread: " << kBasePerThread << "\n"
               << "kpackPerBlock: " << kpackPerBlock << "\n"
               << "bufferA type: " << adaptor.getBufferA().getType() << "\n"
               << "bufferB type: " << adaptor.getBufferB().getType() << "\n");

    // The following loop nest hardcodes the following loop schedule:
    //
    // for(index_t m_i = 0; m_i < mRepeats; ++m_i)
    //   regsA = threadwise_readinto[m_i, :]
    //   for(index_t n_i = 0; n_i<nRepeats; ++n_i)
    //       regsB = threadwise_readint[n_i, :]
    //       threadwise_gemm(regsA, regsB)
    //
    // Which mimics:
    // https://github.com/ROCm/composable_kernel/blob/develop/include/ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp#L304
    //
    // Please note that different schedules might exist, so this can be
    // considered a temporary hack until we have a proper way of "searching"
    // through different schedules (either heuristically or automatically)

    Value wrappedLDSBufferForLoadA = accelEmitterPtr->wrapLDSBufferForLoad(
        b, loc, op.getMatrixA(), op.getBlockSize(), op.getInMPerThread(), "m",
        op.getRotateMWithK());
    Value wrappedLDSBufferForLoadB = accelEmitterPtr->wrapLDSBufferForLoad(
        b, loc, op.getMatrixB(), op.getBlockSize(), op.getInNPerThread(), "n",
        op.getRotateNWithK());

    auto mLoop = b.create<affine::AffineForOp>(loc, 0, mRepeats);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(mLoop.getBody());
      Value i = mLoop.getInductionVar();

      // regsA = read A from LDS
      b.create<ThreadwiseReadIntoOp>(
          loc, wrappedLDSBufferForLoadA, op.getBufferA(), b.getArrayAttr({}),
          ValueRange{tid, i}, /*forceUnroll=*/true, /*useIndexDiffs=*/true);

      auto nLoop = b.create<affine::AffineForOp>(loc, 0, nRepeats);
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(nLoop.getBody());
        Value j = nLoop.getInductionVar();

        // regsB = read B from LDS
        b.create<ThreadwiseReadIntoOp>(
            loc, wrappedLDSBufferForLoadB, op.getBufferB(), b.getArrayAttr({}),
            ValueRange{tid, j}, /*forceUnroll=*/true, /*useIndexDiffs=*/true);

        // regsC += regsA * regsB
        auto kLoop = b.create<affine::AffineForOp>(loc, 0, kBasePerThread);
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(kLoop.getBody());
          Value viewA = accelEmitterPtr->generateThreadwiseViewBufferA(
              b, loc, adaptor.getBufferA());
          Value viewB = accelEmitterPtr->generateThreadwiseViewBufferB(
              b, loc, adaptor.getBufferB());
          Value viewC = accelEmitterPtr->generateThreadwiseViewBufferC(
              b, loc, adaptor.getMatrixC());
          Value k = kLoop.getInductionVar();
          b.create<ThreadwiseAccelGemmOp>(loc, viewA, viewB, viewC,
                                          ValueRange{i, j, k}, arch,
                                          op.getFeaturesAttr(), tuningParams);
        }
      }
    }
    b.eraseOp(op);
    return success();
  }
};

namespace {
struct ThreadwiseReadIntoRewritePattern
    : public OpConversionPattern<ThreadwiseReadIntoOp> {
  using OpConversionPattern<ThreadwiseReadIntoOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ThreadwiseReadIntoOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const final;
};

struct ThreadwiseWriteAllRewritePattern
    : public OpConversionPattern<ThreadwiseWriteAllOp> {
  using OpConversionPattern<ThreadwiseWriteAllOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ThreadwiseWriteAllOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const final;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// BlockwiseReduce lowering.
//===----------------------------------------------------------------------===//

struct BlockwiseReduceRewritePattern
    : public OpConversionPattern<BlockwiseBroadcastReduceOp> {
  using OpConversionPattern<BlockwiseBroadcastReduceOp>::OpConversionPattern;

  int64_t calculateNonReductionDimProduct(ArrayRef<int64_t> toReduceShape,
                                          int64_t axis) const {
    int64_t dimProduct = 1;
    for (size_t i = 0; i < toReduceShape.size(); i++) {
      if (i != (size_t)axis) {
        dimProduct *= toReduceShape[i];
      }
    }
    return dimProduct;
  }

  // This function will make a 2d view from a multi-dimensional tensors
  // where one axis needs to be reduced.
  ArrayAttr createInput2DView(Location loc, PatternRewriter &rewriter,
                              ArrayAttr regTensorView, int64_t reduceAxis,
                              bool makeRDimZero = false) const {
    TransformMapAttr lowestTr =
        cast<TransformMapAttr>(regTensorView[regTensorView.size() - 1]);
    ArrayRef<int64_t> lowestShape = lowestTr.getLowerBounds().asArrayRef();
    // Kreiranje format stringa za ispis
    // auto formatStr = rewriter.getStringAttr("lowestShape: [%lld, %lld]\n");
    // Pripremite tipove rezultata
    /*mlir::Type i64Type = rewriter.getI64Type();
    // Kreiranje konstantnih vrednosti
    auto const0 = rewriter.create<mlir::arith::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(lowestShape[0]));
    auto const1 = rewriter.create<mlir::arith::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(lowestShape[1]));
    // Poziv create za PrintfOp
    rewriter.create<mlir::gpu::PrintfOp>(loc, formatStr,
                                         ValueRange{const0, const1});*/
    TopDownTMBuilder tensorToLDSViewBuilder(rewriter, lowestShape, loc);
    SmallVector<StringRef, 4> upperNameRefs;
    tensorToLDSViewBuilder.getStartNames(upperNameRefs);

    /*std::string joinedNames = llvm::join(upperNameRefs, ", "); //ne radi ne
    moze da se na ispravna nacin pretvori string ref u value
    rewriter.create<mlir::gpu::PrintfOp>(
        loc,
        rewriter.getStringAttr("Gornje reference imena: [%s]\n"),
        ValueRange{joinedNames}
    );*/

    SmallVector<StringRef, 4> nonReduceNameRefs;
    SmallVector<unsigned, 4> nonReduceDims;
    SmallVector<int64_t, 4> nonReduceDimSizes;
    for (auto [dim, dimSize] : llvm::enumerate(lowestShape)) {
      if (dim != (size_t)reduceAxis) {
        nonReduceNameRefs.push_back(upperNameRefs[dim]);
        nonReduceDims.push_back(dim);
        nonReduceDimSizes.push_back(dimSize);
      }
    }
    tensorToLDSViewBuilder.unmerge("nrDim", 0, nonReduceNameRefs,
                                   nonReduceDimSizes);
    if (makeRDimZero) {
      tensorToLDSViewBuilder.constDim("rDim", 1, 0, lowestShape[reduceAxis]);
    } else {
      tensorToLDSViewBuilder.passThrough({"rDim"}, {1},
                                         {upperNameRefs[reduceAxis]});
    }

    TransformMapAttr twoDimLDSView = tensorToLDSViewBuilder.get();
    return prependUpperViews(rewriter, regTensorView,
                             rewriter.getArrayAttr({twoDimLDSView}));
  }

  ArrayAttr create2DToFlatLDSView(Location loc, PatternRewriter &rewriter,
                                  int64_t dim0, int64_t dim1) const {
    TopDownTMBuilder toLDSViewBuilder(rewriter, {dim0, dim1}, loc);
    SmallVector<StringRef, 4> upperNameRefs;
    toLDSViewBuilder.getStartNames(upperNameRefs);
    toLDSViewBuilder.unmerge("flatDim", 0, upperNameRefs, {dim0, dim1});
    return rewriter.getArrayAttr({toLDSViewBuilder.get()});
  }

  // This function will append views to target a flat LDS buffer
  // where non-reduction dims are laid contigously as they are expected
  // function on parallel.
  ArrayAttr createLDSWorkspaceView(
      Location loc, PatternRewriter &rewriter, ArrayAttr regTensorView,
      int64_t reduceAxis, bool makeRDimZero = false,
      std::optional<int64_t> rDimZeroLen = std::nullopt) const {

    TransformMapAttr lowestTr =
        cast<TransformMapAttr>(regTensorView[regTensorView.size() - 1]);
    ArrayRef<int64_t> lowestShape = lowestTr.getLowerBounds().asArrayRef();
    TopDownTMBuilder tensorToLDSViewBuilder(rewriter, lowestShape, loc);
    SmallVector<StringRef, 4> upperNameRefs;
    tensorToLDSViewBuilder.getStartNames(upperNameRefs);
    int64_t rDimLen = rDimZeroLen.value_or(lowestShape[reduceAxis]);

    int64_t nonReduceMergeDimSize = 1;
    SmallVector<StringRef, 4> nonReduceNameRefs;
    SmallVector<unsigned, 4> nonReduceDims;
    SmallVector<int64_t, 4> nonReduceDimSizes;
    for (auto [dim, dimSize] : llvm::enumerate(lowestShape)) {
      if (dim != (size_t)reduceAxis) {
        nonReduceMergeDimSize *= dimSize;
        nonReduceNameRefs.push_back(upperNameRefs[dim]);
        nonReduceDims.push_back(dim);
        nonReduceDimSizes.push_back(dimSize);
      }
    }
    tensorToLDSViewBuilder.unmerge("nrDim", 0, nonReduceNameRefs,
                                   nonReduceDimSizes);
    if (makeRDimZero) {
      tensorToLDSViewBuilder.constDim("rDim", 1, 0, rDimLen);
    } else {
      tensorToLDSViewBuilder.passThrough({"rDim"}, {1},
                                         {upperNameRefs[reduceAxis]});
    }
    TransformMapAttr twoDimLDSView = tensorToLDSViewBuilder.get();

    TopDownTMBuilder flatLDSViewBuilder =
        TopDownTMBuilder::below(tensorToLDSViewBuilder, twoDimLDSView);
    flatLDSViewBuilder.unmerge("flatDim", 0, {"nrDim", "rDim"},
                               {nonReduceMergeDimSize, rDimLen});
    TransformMapAttr flatLDSView = flatLDSViewBuilder.get();
    SmallVector<Attribute> threadsToLDSViewAttrs;
    for (Attribute trMap : regTensorView) {
      threadsToLDSViewAttrs.push_back(trMap);
    }
    threadsToLDSViewAttrs.push_back(twoDimLDSView);
    threadsToLDSViewAttrs.push_back(flatLDSView);
    return rewriter.getArrayAttr(threadsToLDSViewAttrs);
  }

  // This should only be used if product non-reduction dims is
  // equal or larger than number threads in a block.
  //
  // Given a input tensor : D0, ... , Dr , ... , DN to reduce,
  // This function creates a view that maps the space of
  // [D0, ... , Dr , ... , DN] --> [tid, nrIter, rIter] where
  // tid is threads within the block, nrIter is non-reducing
  // iterations within a thread and rIter is reducing iterations
  // within a thread.
  ArrayAttr createThreadViewForNRLargerThanThreads(
      Location loc, ArrayRef<int64_t> toReduceShape, int64_t blockSize,
      int64_t reduceAxis, PatternRewriter &rewriter) const {
    BottomUpTMBuilder threadsToTensor(rewriter, toReduceShape, loc);
    SmallVector<StringRef, 4> lowerNameRefs;
    threadsToTensor.getStartNames(lowerNameRefs);

    int64_t nonReduceMergeDimSize = 1;
    SmallVector<StringRef, 4> nonReduceNameRefs;
    for (auto dimAndSize : llvm::enumerate(toReduceShape)) {
      int64_t dim = dimAndSize.index();
      int64_t dimSize = dimAndSize.value();
      if (dim != reduceAxis) {
        nonReduceMergeDimSize *= dimSize;
        nonReduceNameRefs.push_back(lowerNameRefs[dim]);
      }
    }
    threadsToTensor.merge("nrDim", 0, nonReduceNameRefs);
    threadsToTensor.passThrough({"rIter"}, {1}, {lowerNameRefs[reduceAxis]});
    TransformMapAttr mergeTrMap = threadsToTensor.get();

    threadsToTensor = BottomUpTMBuilder::above(threadsToTensor, mergeTrMap);
    int64_t nrThreads = (nonReduceMergeDimSize + (blockSize - 1)) / blockSize;
    threadsToTensor.pad({"nrDim"},
                        {0, blockSize * nrThreads - nonReduceMergeDimSize});
    threadsToTensor.passThrough({"rIter"}, {1}, {"rIter"});
    TransformMapAttr padTrMap = threadsToTensor.get();

    threadsToTensor = BottomUpTMBuilder::above(threadsToTensor, padTrMap);
    threadsToTensor.unmerge({"tid", "nrIter"}, {0, 1}, "nrDim",
                            {blockSize, nrThreads});
    threadsToTensor.passThrough({"rIter"}, {2}, {"rIter"});
    TransformMapAttr unmergeTrMap = threadsToTensor.get();

    return rewriter.getArrayAttr({unmergeTrMap, padTrMap, mergeTrMap});
  }

  // This should only be used if product non-reduction dims is
  // less than number threads in a block.
  //
  // Given a input tensor : D0, ... , Dr , ... , DN to reduce,
  // This function creates a view that maps the space of
  // [D0, ... , Dr , ... , DN] --> [nrtid, rtid, rIter] where
  // nrtid = tid / product(non-reduction dims) is a reduction subgroup leader.
  // rtid = tid % product(non-reduction dims) is thread idx within a reduction
  // subgroup. Size of the dimension 'rtid' is the number of threads
  // that'd participate in the reduction
  ArrayAttr createThreadViewforNRSmallerThanThreads(
      Location loc, ArrayRef<int64_t> toReduceShape, int64_t blockSize,
      size_t reduceAxis, PatternRewriter &rewriter) const {
    // rewriter.create<gpu::PrintfOp>(loc, "Initial blockSize: %ld\n",
    // ValueRange{rewriter.create<mlir::arith::ConstantIntOp>(loc, blockSize,
    // 64)});
    BottomUpTMBuilder threadsToTensor(rewriter, toReduceShape, loc);
    SmallVector<StringRef, 4> lowerNameRefs;
    threadsToTensor.getStartNames(lowerNameRefs);
    int64_t nonReduceMergeDimSize = 1;
    SmallVector<StringRef, 4> nonReduceNameRefs;
    for (auto [dim, dimSize] : llvm::enumerate(toReduceShape)) {
      if (dim != reduceAxis) {
        nonReduceMergeDimSize *= dimSize;
        nonReduceNameRefs.push_back(lowerNameRefs[dim]);
      }
    }

    /*rewriter.create<gpu::PrintfOp>(
        loc, "Non-reduce merge dim size: %ld\n",
        ValueRange{rewriter.create<arith::ConstantIndexOp>(
            loc, nonReduceMergeDimSize)});*/
    threadsToTensor.merge("nrDim", 0, nonReduceNameRefs);
    threadsToTensor.passThrough({"rDim"}, {1}, {lowerNameRefs[reduceAxis]});
    TransformMapAttr mergeTrMap = threadsToTensor.get();

    threadsToTensor = BottomUpTMBuilder::above(threadsToTensor, mergeTrMap);
    // If this function is being called, then the number of threads is larger
    // than the product of non reduction dimensions. Therefore, we create thread
    // groups (rthreads) per a point in merge(non reduction dimensions).
    int64_t rthreads = blockSize / nonReduceMergeDimSize;
    /*rewriter.create<gpu::PrintfOp>(
        loc, "RThreads: %ld\n",
        ValueRange{rewriter.create<arith::ConstantIndexOp>(loc, rthreads)});*/
    int64_t rDimPerRThread =
        (toReduceShape[reduceAxis] + (rthreads - 1)) / rthreads;
    /*rewriter.create<gpu::PrintfOp>(
        loc, "rDimPerRThread: %ld\n",
        ValueRange{
            rewriter.create<arith::ConstantIndexOp>(loc, rDimPerRThread)});*/
    threadsToTensor.pad(
        {"rDim"}, {0, rthreads * rDimPerRThread - toReduceShape[reduceAxis]});
    threadsToTensor.passThrough({"nrDim"}, {0}, {"nrDim"});
    TransformMapAttr padTrMap = threadsToTensor.get();

    threadsToTensor = BottomUpTMBuilder::above(threadsToTensor, padTrMap);
    threadsToTensor.unmerge({"rtid", "rIter"}, {1, 2}, "rDim",
                            {rthreads, rDimPerRThread});
    threadsToTensor.passThrough({"nrtid"}, {0}, {"nrDim"});
    TransformMapAttr unmergeTrMap = threadsToTensor.get();

    return rewriter.getArrayAttr({unmergeTrMap, padTrMap, mergeTrMap});
  }

  Value getReductionInitValue(BlockwiseBroadcastReduceOp op,
                              ConversionPatternRewriter &rewriter) const {
    ReduceMethod rMethod = op.getReduceMethod();
    Type elementType = op.getInput().getType().getElementType();
    if (elementType.isIntOrIndex()) {
      if (rMethod == ReduceMethod::Sum) {
        return createConstantIntOp(rewriter, op.getLoc(), elementType,
                                   elementType, 0);
      } else {
        // Op verifier gurantees this.
        assert(rMethod == ReduceMethod::Max);
        return createConstantIntOp(rewriter, op.getLoc(), elementType,
                                   elementType,
                                   std::numeric_limits<int64_t>::min());
      }
    } else {
      if (rMethod == ReduceMethod::Sum) {
        return createConstantFloatOp(rewriter, op.getLoc(), elementType,
                                     elementType, 0.0);
      } else {
        // Op verifier gurantees this.
        assert(rMethod == ReduceMethod::Max);
        return createConstantFloatOp(rewriter, op.getLoc(), elementType,
                                     elementType,
                                     -std::numeric_limits<float>::infinity());
      }
    }
  }

  void printInput(Value input, OpBuilder &builder, Location loc) const {
    // Format string za ispis
    auto formatString = builder.getStringAttr("Input value: %f\n");

    // Proverite tip inputa
    Value inputToPrint = input;
    /*if (isa<mlir::TensorType>(input.getType())) {
        // Uzmite prvi element ako je tensor
        inputToPrint = builder.create<mlir::arith::ExtractElementOp>(loc, input,
    builder.getI64ArrayAttr({0}));
    }*/

    // Kreirajte gpu.printf
    builder.create<mlir::gpu::PrintfOp>(loc, formatString, inputToPrint);
  }

  Value createReducingOp(BlockwiseBroadcastReduceOp op, Value input, Value acc,
                         OpBuilder &builder) const {
    ReduceMethod rMethod = op.getReduceMethod();
    Location loc = op.getLoc();
    // Value loadAcc = rewriter.create<InBoundsLoadOp>(loc, input.getType(),
    // acc, zeroConstantOp);
    Type elementType = op.getInput().getType().getElementType();

    // Debug: Ispiši vrednost input
    // printInput(input, builder, loc);

    if (!isa<VectorType>(acc.getType()) && isa<VectorType>(input.getType())) {
      // This means accumulator is a scalar type and input is a vector type,
      // therefore its a elementwise reduction between two operands.
      vector::CombiningKind kind;
      if (rMethod == ReduceMethod::Sum) {
        kind = vector::CombiningKind::ADD;
      } else {
        // Op verifier gurantees this.
        assert(rMethod == ReduceMethod::Max);
        if (elementType.isIntOrIndex()) {
          kind = vector::CombiningKind::MAXIMUMF;
        } else {
          kind = vector::CombiningKind::MAXIMUMF;
        }
      }
      input = builder.create<vector::ReductionOp>(loc, kind, input);
    }

    if (rMethod == ReduceMethod::Sum) {
      Value reduced;
      if (elementType.isIntOrIndex()) {
        reduced = builder.create<arith::AddIOp>(loc, acc, input);
      } else {
        reduced = builder.create<arith::AddFOp>(loc, acc, input);
      }
      return reduced;
    } else {
      assert(rMethod == ReduceMethod::Max);
      Value reduced;
      if (elementType.isIntOrIndex()) {
        reduced = builder.create<arith::MaxSIOp>(loc, acc, input);
      } else {
        reduced = builder.create<arith::MaximumFOp>(loc, acc, input);
      }
      return reduced;
    }
  }

  ArrayAttr createReducedView(PatternRewriter &rewriter, Location loc,
                              ArrayAttr subTileView, int64_t axis) const {
    ArrayRef<int64_t> threadSubTileShape = getLowerShape(subTileView);
    TopDownTMBuilder viewBuilder(rewriter, threadSubTileShape, loc);
    for (auto [dim, dimSize] : llvm::enumerate(threadSubTileShape)) {
      if ((int64_t)dim == axis) {
        viewBuilder.constDim("rDim", dim, 0, dimSize);
      } else {
        viewBuilder.passThrough({(unsigned int)dim}, {(unsigned int)dim});
      }
    }
    TransformMapAttr redDimZeroMap = viewBuilder.get();
    ArrayAttr reducedView = prependUpperViews(
        rewriter, subTileView, rewriter.getArrayAttr({redDimZeroMap}));
    return reducedView;
  }

  // Perform threadwise reductions based thread subtile
  // view and store the reduced data to reduced buffer
  void doThreadwiseReductions(PatternRewriter &rewriter, Location loc,
                              BlockwiseBroadcastReduceOp op,
                              Value reducedBuffer, int64_t blockSize,
                              ArrayAttr inputThreadSubTile2dView) const {
    Value inputRawBuffer = op.getInput();
    // auto srcBufferType = cast<MemRefType>(inputRawBuffer.getType());
    // llvm::errs() << "Izlaz: " <<
    // cast<gpu::AddressSpaceAttr>(srcBufferType.getMemorySpace()).getValue() <<
    // "\n"; llvm::errs() << "Tip" << reducedBuffer.getType() << "\n";
    WorkitemIdOp tid =
        rewriter.create<WorkitemIdOp>(loc, rewriter.getIndexType());
    WorkgroupIdOp blockId =
        rewriter.create<WorkgroupIdOp>(loc, rewriter.getIndexType());
    int64_t numElements =
        cast<MemRefType>(inputRawBuffer.getType()).getNumElements();
    /*rewriter.create<mlir::gpu::PrintfOp>(
        loc,
        rewriter.getStringAttr(
            "Threadwise reduction: Number of elements: %d\n"),
        ValueRange{rewriter.create<arith::ConstantIndexOp>(loc, numElements)});*/
    constexpr size_t nrDim = 0;

    Type elemType = cast<MemRefType>(inputRawBuffer.getType()).getElementType();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    auto loop = rewriter.create<TransformingForOp>(
        loc, ArrayRef<ValueRange>{{zero}, {zero}},
        ArrayRef<Attribute>{inputThreadSubTile2dView,
                            rewriter.getArrayAttr({})},
        /*bounds=*/ArrayRef<int64_t>{numElements},
        /*strides=*/ArrayRef<int64_t>{1},
        /*useIndexDiffs=*/true, /*forceUnroll=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(loop.getBody());
      Block::BlockArgListType upperCoords = loop.getLowerCoords(1);
      Block::BlockArgListType subtileCoords = loop.getLowerCoords(0);

      Value ldInput = rewriter.create<InBoundsLoadOp>(
          loc, elemType, inputRawBuffer, upperCoords);
      /*rewriter.create<mlir::gpu::PrintfOp>(
          loc,
          rewriter.getStringAttr("ThreadwiseReduction: block: %d "
                                 "tid: %d ldInput: %f\n"),
          ValueRange{blockId, tid, ldInput});*/
      Value ldInputAcc = rewriter.create<InBoundsLoadOp>(
          loc, elemType, reducedBuffer, subtileCoords[nrDim]);

      // Poziv ROCDL_SetInactiveOp
      /*Value setInactiveValue = rewriter.create<ROCDL::SetInactiveOp>(
          loc, elemType, ldInput, ldInputAcc);
      // Dodaj printf za provere
      rewriter.create<mlir::gpu::PrintfOp>(
          loc,
          rewriter.getStringAttr("ThreadwiseReduction: block: %d tid: %d "
                                 "ldInput: %f SetInactiveValue: %f\n"),
          ValueRange{blockId, tid, ldInput, setInactiveValue});

      Value dppResult1 = rewriter.create<ROCDL::DPPUpdateOp>(
          loc, elemType, setInactiveValue, setInactiveValue, 0x110 + 1, 0xF,
          0xF, false);

      Value dppResult = createReducingOp(op, setInactiveValue, dppResult1,
      rewriter); rewriter.create<mlir::gpu::PrintfOp>( loc,
          rewriter.getStringAttr(
              "ThreadwiseReduction: block: %d tid: %d dppResult "
              "after DPP1111111111111: %f\n"),
          ValueRange{blockId, tid, dppResult});

      Value dppResult2 = rewriter.create<ROCDL::DPPUpdateOp>(
          loc, elemType, setInactiveValue, setInactiveValue, 0x110 + 2, 0xF,
      0xF, false);

      dppResult = createReducingOp(op, dppResult, dppResult2, rewriter);
      rewriter.create<mlir::gpu::PrintfOp>(
          loc,
          rewriter.getStringAttr(
              "ThreadwiseReduction: block: %d tid: %d dppResult "
              "after DPP2222222222: %f\n"),
          ValueRange{blockId, tid, dppResult});

      Value dppResult3 = rewriter.create<ROCDL::DPPUpdateOp>(
            loc, elemType, setInactiveValue, setInactiveValue, 0x110 + 3, 0xF,
            0xF, false);

        dppResult = createReducingOp(op, dppResult, dppResult3, rewriter);
        rewriter.create<mlir::gpu::PrintfOp>(
            loc,
            rewriter.getStringAttr(
                "storePartialReductionstoLDS: block: %d tid: %d dppResult "
                "after DPP333333333: %f\n"),
            ValueRange{blockId, tid, dppResult});

      Value dppResult4 = rewriter.create<ROCDL::DPPUpdateOp>(
          loc, elemType, dppResult, dppResult, 0x110 + 4, 0xF, 0xE, false);

      dppResult = createReducingOp(op, dppResult, dppResult4, rewriter);

      rewriter.create<mlir::gpu::PrintfOp>(
          loc,
          rewriter.getStringAttr(
              "ThreadwiseReduction: block: %d tid: %d dppResult "
              "after DPP444444444444444: %f\n"),
          ValueRange{blockId, tid, dppResult});

      Value dppResult5 = rewriter.create<ROCDL::DPPUpdateOp>(
          loc, elemType, dppResult, dppResult, 0x110 + 8, 0xF, 0xC, false);

      dppResult = createReducingOp(op, dppResult, dppResult5, rewriter);
      rewriter.create<mlir::gpu::PrintfOp>(
          loc,
          rewriter.getStringAttr(
              "ThreadwiseReduction: block: %d tid: %d dppResult "
              "after DPP88888888888888: %f\n"),
          ValueRange{blockId, tid, dppResult});

      Value dppBrodcast = rewriter.create<ROCDL::DPPUpdateOp>(
          loc, elemType, dppResult, dppResult, 0x142, 0xA, 0xF, false);

      dppResult = createReducingOp(op, dppResult, dppBrodcast, rewriter);
      rewriter.create<mlir::gpu::PrintfOp>(
          loc,
          rewriter.getStringAttr("ThreadwiseReduction: block: %d "
                                 "tid: %d dppResult after DPPbcast15: %f\n"),
          ValueRange{blockId, tid, dppResult});

      dppBrodcast = rewriter.create<ROCDL::DPPUpdateOp>(
          loc, elemType, dppResult, dppResult, 0x143, 0xC, 0xF, false);

      dppResult = createReducingOp(op, dppResult, dppBrodcast, rewriter);
      rewriter.create<mlir::gpu::PrintfOp>(
          loc,
          rewriter.getStringAttr("ThreadwiseReduction: block: %d "
                                 "tid: %d dppResult after DPPbcast31: %f\n"),
          ValueRange{blockId, tid, dppResult});

      Value dppRotated = rewriter.create<ROCDL::DPPUpdateOp>(
          loc, elemType, dppResult, dppResult, 0x13C, 0xF, 0xF, false);

      rewriter.create<mlir::gpu::PrintfOp>(
          loc,
          rewriter.getStringAttr("ThreadwiseReduction: block: %d "
                                 "tid: %d dppResult after DPProtate: %f\n"),
          ValueRange{blockId, tid, dppRotated});

      dppRotated =
          rewriter.create<ROCDL::StrictWWMOp>(loc, elemType, dppRotated);

      rewriter.create<mlir::gpu::PrintfOp>(
          loc,
          rewriter.getStringAttr("ThreadwiseReduction: block: %d "
                                 "tid: %d dppResult after WWM: %f\n"),
          ValueRange{blockId, tid, dppRotated});*/

      /*wwmresult = rewriter.create<ROCDL::StrictWQMOp>(
     loc, elemType, dppRotated);

     rewriter.create<mlir::gpu::PrintfOp>(
          loc,
          rewriter.getStringAttr("ThreadwiseReduction: block: %d "
                                 "tid: %d dppResult after WWM: %f\n"),
          ValueRange{blockId, tid, wwmresult});

      rewriter.create<InBoundsStoreOp>(loc, wwmresult, reducedBuffer,
                                       subtileCoords[nrDim]);*/

      Value reduced = createReducingOp(op, ldInput, ldInputAcc, rewriter);
      /*rewriter.create<gpu::PrintfOp>(
          loc,
          rewriter.getStringAttr(
              "Threadwise reduction: tid: %d Reduced Value: %f\n"),
          ValueRange{tid, reduced});*/
      rewriter.create<InBoundsStoreOp>(loc, reduced, reducedBuffer,
                                       subtileCoords[nrDim]);
    }
  }

  // This function store partial reductions to LDS for
  // inter-thread reductions later on.
  void storePartialReductionstoLDS(
      BlockwiseBroadcastReduceOp op, PatternRewriter &rewriter, Location loc,
      Value reducedBuffer, Value ldsBuffer, ArrayAttr inputBlockSubTile2dView,
      ArrayAttr inputThreadSubTile2dView, ArrayAttr tidSubTileSliceView,
      ArrayAttr toFlatLDSView, int64_t blockSize) const {
    Type elemType = cast<MemRefType>(reducedBuffer.getType()).getElementType();
    constexpr size_t nrDim = 0;
    constexpr size_t rDim = 1;
    ArrayAttr inputThreadSubTile2dViewInv =
        invertTransforms(rewriter, loc, inputThreadSubTile2dView);
    ArrayRef<int64_t> threadSubTile2DShape =
        getLowerShape(inputThreadSubTile2dView);
    WorkitemIdOp tid =
        rewriter.create<WorkitemIdOp>(loc, rewriter.getIndexType());
    WorkgroupIdOp blockId =
        rewriter.create<WorkgroupIdOp>(loc, rewriter.getIndexType());
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    // First we iterate thread subtile along non-reduction
    // axis to get iter coordinate within the register
    auto loop = rewriter.create<TransformingForOp>(
        loc, ArrayRef<ValueRange>{{zero, zero}, {zero, zero}},
        ArrayRef<Attribute>{inputThreadSubTile2dViewInv,
                            rewriter.getArrayAttr({})},
        /*bounds=*/ArrayRef<int64_t>{threadSubTile2DShape[nrDim], 1},
        /*strides=*/ArrayRef<int64_t>{1, 1},
        /*useIndexDiffs=*/true, /*forceUnroll=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(loop.getBody());
      Value iter = loop.getLowerCoords(0)[0];
      Block::BlockArgListType threadSubTile2DCoords = loop.getLowerCoords(1);

      // Then we plug that iter coordinate along with tid to recover block
      // subtile coordinates. However, we only need non-reduction dimension
      // coordinate from the block subtile.
      auto convertToBlockSubTile = rewriter.create<TransformingForOp>(
          loc, ArrayRef<ValueRange>{{tid, iter}},
          ArrayRef<Attribute>{inputBlockSubTile2dView},
          /*bounds=*/ArrayRef<int64_t>{1, 1},
          /*strides=*/ArrayRef<int64_t>{1, 1},
          /*useIndexDiffs=*/true, /*forceUnroll=*/true);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(convertToBlockSubTile.getBody());
        Value blockNrDimCoord = convertToBlockSubTile.getLowerCoords(
            0)[nrDim]; // blockNrDimCoord: 0.000000
        Value ldReduced = rewriter.create<InBoundsLoadOp>(
            loc, elemType, reducedBuffer, ValueRange{threadSubTile2DCoords[0]});
        /*rewriter.create<mlir::gpu::PrintfOp>(
            loc,
            rewriter.getStringAttr("storePartialReductionstoLDS: block: %d "
                                   "tid: %d ldReduced: %f\n"),
            ValueRange{blockId, tid, ldReduced});*/
        // Value laneId = tid.getResult(); // Lane ID (0-63)

        Value inactiveValue = rewriter.create<arith::ConstantOp>(
            loc, elemType, rewriter.getFloatAttr(elemType, 0.0));

        Value setInactiveValue = rewriter.create<ROCDL::SetInactiveOp>(
            loc, elemType, ldReduced, inactiveValue);
        // Dodaj printf za provere
        /*rewriter.create<mlir::gpu::PrintfOp>(
            loc,
            rewriter.getStringAttr("ThreadwiseReduction: block: %d tid: %d "
                                   "ldInput: %f SetInactiveValue: %f\n"),
            ValueRange{blockId, tid, ldReduced, setInactiveValue});*/

        Value dppResult1 = rewriter.create<amdgpu::DPPOp>(
            loc, elemType, setInactiveValue, setInactiveValue,
            amdgpu::DPPPermAttr::get(rewriter.getContext(),
                                     amdgpu::DPPPerm::row_shr),
            rewriter.getI32IntegerAttr(1), rewriter.getI32IntegerAttr(0xF),
            rewriter.getI32IntegerAttr(0xF), rewriter.getBoolAttr(false));

        Value dppResult =
            createReducingOp(op, setInactiveValue, dppResult1, rewriter);
        /*rewriter.create<mlir::gpu::PrintfOp>(
            loc,
            rewriter.getStringAttr(
                "storePartialReductionstoLDS: block: %d tid: %d dppResult "
                "after DPP1111111111111: %f\n"),
            ValueRange{blockId, tid, dppResult});*/

        Value dppResult2 = rewriter.create<amdgpu::DPPOp>(
            loc, elemType, setInactiveValue, setInactiveValue, amdgpu::DPPPermAttr::get(rewriter.getContext(),
                                     amdgpu::DPPPerm::row_shr),
            rewriter.getI32IntegerAttr(2), rewriter.getI32IntegerAttr(0xF),
            rewriter.getI32IntegerAttr(0xF), rewriter.getBoolAttr(false));

        dppResult = createReducingOp(op, dppResult, dppResult2, rewriter);
        /*rewriter.create<mlir::gpu::PrintfOp>(
            loc,
            rewriter.getStringAttr(
                "storePartialReductionstoLDS: block: %d tid: %d dppResult "
                "after DPP2222222222: %f\n"),
            ValueRange{blockId, tid, dppResult});*/

        Value dppResult3 = rewriter.create<amdgpu::DPPOp>(
            loc, elemType, setInactiveValue, setInactiveValue, amdgpu::DPPPermAttr::get(rewriter.getContext(),
                                     amdgpu::DPPPerm::row_shr),
            rewriter.getI32IntegerAttr(3), rewriter.getI32IntegerAttr(0xF),
            rewriter.getI32IntegerAttr(0xF), rewriter.getBoolAttr(false));

        dppResult = createReducingOp(op, dppResult, dppResult3, rewriter);
        /*rewriter.create<mlir::gpu::PrintfOp>(
            loc,
            rewriter.getStringAttr(
                "storePartialReductionstoLDS: block: %d tid: %d dppResult "
                "after DPP333333333: %f\n"),
            ValueRange{blockId, tid, dppResult});*/

        Value dppResult4 = rewriter.create<amdgpu::DPPOp>(
            loc, elemType, dppResult, dppResult, amdgpu::DPPPermAttr::get(rewriter.getContext(),
                                     amdgpu::DPPPerm::row_shr),
            rewriter.getI32IntegerAttr(4), rewriter.getI32IntegerAttr(0xF),
            rewriter.getI32IntegerAttr(0xE), rewriter.getBoolAttr(false));

        dppResult = createReducingOp(op, dppResult, dppResult4, rewriter);

        /*rewriter.create<mlir::gpu::PrintfOp>(
            loc,
            rewriter.getStringAttr(
                "storePartialReductionstoLDS: block: %d tid: %d dppResult "
                "after DPP444444444444444: %f\n"),
            ValueRange{blockId, tid, dppResult});*/
        Value dppResult5 = rewriter.create<amdgpu::DPPOp>(
            loc, elemType, dppResult, dppResult, amdgpu::DPPPermAttr::get(rewriter.getContext(),
                                     amdgpu::DPPPerm::row_shr),
            rewriter.getI32IntegerAttr(8), rewriter.getI32IntegerAttr(0xF),
            rewriter.getI32IntegerAttr(0xC), rewriter.getBoolAttr(false));

        dppResult = createReducingOp(op, dppResult, dppResult5, rewriter);
        /*rewriter.create<mlir::gpu::PrintfOp>(
            loc,
            rewriter.getStringAttr(
                "storePartialReductionstoLDS: block: %d tid: %d dppResult "
                "after DPP88888888888888: %f\n"),
            ValueRange{blockId, tid, dppResult});*/

        Value dppBrodcast = rewriter.create<amdgpu::DPPOp>(
            loc, elemType, dppResult, dppResult, amdgpu::DPPPermAttr::get(rewriter.getContext(),
                                     amdgpu::DPPPerm::row_bcast_15),
            nullptr, rewriter.getI32IntegerAttr(0xA),
            rewriter.getI32IntegerAttr(0xF), rewriter.getBoolAttr(false));

        dppResult = createReducingOp(op, dppResult, dppBrodcast, rewriter);
        /*rewriter.create<mlir::gpu::PrintfOp>(
            loc,
            rewriter.getStringAttr("storePartialReductionstoLDS: block: %d "
                                   "tid: %d dppResult after DPPbcast15: %f\n"),
            ValueRange{blockId, tid, dppResult});*/

        dppBrodcast = rewriter.create<amdgpu::DPPOp>(
            loc, elemType, dppResult, dppResult, amdgpu::DPPPermAttr::get(rewriter.getContext(),
                                     amdgpu::DPPPerm::row_bcast_31),
             nullptr, rewriter.getI32IntegerAttr(0xC),
            rewriter.getI32IntegerAttr(0xF), rewriter.getBoolAttr(false));

        dppResult = createReducingOp(op, dppResult, dppBrodcast, rewriter);
        /*rewriter.create<mlir::gpu::PrintfOp>(
            loc,
            rewriter.getStringAttr("storePartialReductionstoLDS: block: %d "
                                   "tid: %d dppResult after DPPbcast31: %f\n"),
            ValueRange{blockId, tid, dppResult});*/

        Value dppRotated = rewriter.create<amdgpu::DPPOp>(
            loc, elemType, dppResult, dppResult, amdgpu::DPPPermAttr::get(rewriter.getContext(),
                                     amdgpu::DPPPerm::wave_ror),
            nullptr, rewriter.getI32IntegerAttr(0xF),
            rewriter.getI32IntegerAttr(0xF), rewriter.getBoolAttr(false));

        /*rewriter.create<mlir::gpu::PrintfOp>(
            loc,
            rewriter.getStringAttr("storePartialReductionstoLDS: block: %d "
                                   "tid: %d dppResult after DPProtate: %f\n"),
            ValueRange{blockId, tid, dppRotated});*/

        dppRotated =
            rewriter.create<ROCDL::StrictWWMOp>(loc, elemType, dppRotated);

        /*rewriter.create<mlir::gpu::PrintfOp>(
            loc,
            rewriter.getStringAttr("ThreadwiseReduction: block: %d "
                                   "tid: %d dppResult after WWM: %f\n"),
            ValueRange{blockId, tid, dppRotated});*/

        // Here we plug the tid to get the sliced block subtile coordinate
        // find a unique packed coordinate in the reduction axis per each
        // thread to write the partial reductions to the lds.
        auto convertToBlockSubTileTidSlice = rewriter.create<TransformingForOp>(
            loc, ArrayRef<ValueRange>{{tid}},
            ArrayRef<Attribute>{tidSubTileSliceView},
            /*bounds=*/ArrayRef<int64_t>{1},
            /*strides=*/ArrayRef<int64_t>{1},
            /*useIndexDiffs=*/true, /*forceUnroll=*/true);
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(
              convertToBlockSubTileTidSlice.getBody());
          Value blockTidSliceRDimCoord =
              convertToBlockSubTileTidSlice.getLowerCoords(0)[rDim];
          auto ldsStoreloop = rewriter.create<TransformingForOp>(
              loc,
              ArrayRef<ValueRange>{{blockNrDimCoord, blockTidSliceRDimCoord}},
              ArrayRef<Attribute>{toFlatLDSView},
              /*bounds=*/ArrayRef<int64_t>{1, 1},
              /*strides=*/ArrayRef<int64_t>{1, 1},
              /*useIndexDiffs=*/true, /*forceUnroll=*/true);
          {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(ldsStoreloop.getBody());
            Block::BlockArgListType ldsFlatCoords =
                ldsStoreloop.getLowerCoords(0);
            Value flatCoordX = ldsFlatCoords[0];
            //Value flatCoordY = ldsFlatCoords[1];
            /*rewriter.create<mlir::gpu::PrintfOp>(
            loc,
            rewriter.getStringAttr("InboundsStoreOp: block: %d, tid: %d, ldsFlatCoordX: %d, value: %f\n"),  
                ValueRange{blockId, tid, flatCoordX, dppRotated}); */
            rewriter.create<InBoundsStoreOp>(loc, dppRotated, ldsBuffer,
                                             ldsFlatCoords);
          }
        }
      }
    }
  }

  LogicalResult
  matchAndRewrite(BlockwiseBroadcastReduceOp op,
                  BlockwiseBroadcastReduceOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    // inputView should be register {bid, tid, iter} to virtual tensor {bid,
    // d0,
    // ... , Dr , ... , dn} coords transforms where Dr is the reduction axis.
    ArrayAttr inputViewArrayAttr = op.getInputRegViewAttr();

    TypedValue<MemRefType> inputReg = op.getInput();
    /*rewriter.create<mlir::gpu::PrintfOp>(
        loc, rewriter.getStringAttr("inputReg: %p\n"),
       ValueRange{inputReg});*/
    TypedValue<MemRefType> outputReg = op.getOutput();
    /*rewriter.create<mlir::gpu::PrintfOp>(
        loc, rewriter.getStringAttr("Output: %p\n"), ValueRange{outputReg});*/
    Type elemType = inputReg.getType().getElementType();
    TypedValue<MemRefType> workspaceLDSBuffer = op.getWorkspaceBuffer();
    /*rewriter.create<mlir::gpu::PrintfOp>(
        loc, rewriter.getStringAttr("workspaceLDSBuffer: %p\n"),
        ValueRange{workspaceLDSBuffer});*/
    Value zeroConstantOp = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    int64_t axis = op.getAxis().getSExtValue(); // 1
    /*rewriter.create<mlir::gpu::PrintfOp>(
        loc, rewriter.getStringAttr("axis: %d\n"),
        ValueRange{rewriter.create<mlir::arith::ConstantIntOp>(loc, axis,
       64)});*/
    int64_t blockSize = op.getBlockSize();
    /*rewriter.create<mlir::gpu::PrintfOp>(
        loc, rewriter.getStringAttr("blockSize: %d\n"),
        ValueRange{
            rewriter.create<mlir::arith::ConstantIntOp>(loc, blockSize, 64)});*/
    auto privateMemoryAddressSpace = rewriter.getAttr<gpu::AddressSpaceAttr>(
        gpu::GPUDialect::getPrivateAddressSpace());
    // Dobijamo celobrojnu vrednost adresnog prostora
    /*int64_t addressSpaceValue =
        static_cast<int64_t>(privateMemoryAddressSpace.getValue());
    auto addressSpaceConstantOp = rewriter.create<mlir::arith::ConstantIntOp>(
        loc, addressSpaceValue, rewriter.getIntegerType(64));
    rewriter.create<mlir::gpu::PrintfOp>(
        loc, rewriter.getStringAttr("privateMemoryAddressSpace: %d\n"),
        ValueRange{addressSpaceConstantOp});*/
    // Get current workitem ID.
    WorkitemIdOp tid =
        rewriter.create<WorkitemIdOp>(loc, rewriter.getIndexType());
    /*rewriter.create<mlir::gpu::PrintfOp>(
        loc, rewriter.getStringAttr("tid: %d\n"), ValueRange{tid});*/
    WorkgroupIdOp blockId =
        rewriter.create<WorkgroupIdOp>(loc, rewriter.getIndexType());

    // Create strides and bounds to iterate the virtual tensor
    TransformMapAttr lowerTr = cast<TransformMapAttr>(
        inputViewArrayAttr[inputViewArrayAttr.size() - 1]);
    ArrayRef<int64_t> lowerTrLowerBounds = // tr_map bounds [4,20] 0=4, 1=20
        lowerTr.getLowerBounds().asArrayRef();
    SmallVector<int64_t, 4> regTensorShape =
        llvm::to_vector<4>(lowerTrLowerBounds); // isto kao i prethodni iste
                                                // vrednosti jer se kopiraju

    /*for (size_t i = 0; i < lowerTrLowerBounds.size(); ++i) {
      Value idxValue = rewriter.create<arith::ConstantIndexOp>(loc, i);
      Value lowerBoundValue =
          rewriter.create<arith::ConstantIndexOp>(loc, lowerTrLowerBounds[i]);
      rewriter.create<mlir::gpu::PrintfOp>(
          loc, rewriter.getStringAttr("lowerTrLowerBounds[%d] = %d\n"),
          ValueRange{idxValue, lowerBoundValue});
    }*/

    // 2DView is alwasy nrDim x rdim
    constexpr size_t nrDim = 0;
    constexpr size_t rDim = 1;
    ArrayAttr inputThreadSubTile2dView =
        createInput2DView(loc, rewriter, op.getIterSubTileSliceView(), axis);
    ArrayRef<int64_t> inputThreadSubTile2dShape =
        getLowerShape(inputThreadSubTile2dView);

    /*for (size_t i = 0; i < inputThreadSubTile2dShape.size();
         ++i) { // prvo je 4 iz lowest dhape drugo je 1
      Value idxValue = rewriter.create<arith::ConstantIndexOp>(loc, i);
      Value lowerBoundValue = rewriter.create<arith::ConstantIndexOp>(
          loc, inputThreadSubTile2dShape[i]);
      rewriter.create<mlir::gpu::PrintfOp>(
          loc, rewriter.getStringAttr("inputThreadSubTile2dShape[%d] = %d\n"),
          ValueRange{idxValue, lowerBoundValue});
    }
    Value idxValue = rewriter.create<arith::ConstantIndexOp>(loc,
inputThreadSubTile2dShape[0]); Value lowerBoundValue =
rewriter.create<arith::ConstantIndexOp>(loc, inputThreadSubTile2dShape[1]);
    rewriter.create<gpu::PrintfOp>(
    loc,
    rewriter.getStringAttr("Thread SubTile 2D Shape: [%d,
%d]\n"),ValueRange{idxValue,lowerBoundValue});*/ //isto prikazuje sto i gore
    /*auto wavefrontSize = 64;
    // Kreiranje tipa za buffer sa veličinom za ceo wavefront
    auto partialReductionBufferType1 =
        MemRefType::get({wavefrontSize}, elemType, AffineMap{},
    privateMemoryAddressSpace);
        // Alokacija buffera u privatnom adresnom prostoru
    Value partialReductionBuffer1 =
        rewriter.create<GpuAllocOp>(loc, partialReductionBufferType1);
        // Dobijanje inicijalne vrednosti redukcije
    //Value initVal1 = getReductionInitValue(op, rewriter);
    Value inactiveValue = rewriter.create<arith::ConstantOp>(
              loc, elemType, rewriter.getFloatAttr(elemType, 0.0));
    // Inicijalizacija buffera
    rewriter.create<FillOp>(loc, partialReductionBuffer1, inactiveValue);*/
    auto partialReductionBufferType =
        MemRefType::get(inputThreadSubTile2dShape[nrDim], elemType, AffineMap{},
                        privateMemoryAddressSpace);
    Value partialReductionBuffer = // alocira buffer za vrednosti redukcije
        rewriter.create<GpuAllocOp>(loc, partialReductionBufferType);
    Value initVal = getReductionInitValue(op, rewriter);
    rewriter.create<FillOp>(
        loc, partialReductionBuffer,
        initVal); // ubacuje inicijalne vrednosti iz initvalue funkcije

    doThreadwiseReductions(rewriter, loc, op, partialReductionBuffer, blockSize,
                           inputThreadSubTile2dView);

    // Create partially reduced tensor shape
    ArrayAttr inputBlockSubTile2dView =
        createInput2DView(loc, rewriter, inputViewArrayAttr, axis);
    SmallVector<int64_t, 2> partialRegTensorShape =
        llvm::to_vector<2>(getLowerShape(inputBlockSubTile2dView));
    /*rewriter.create<gpu::PrintfOp>(
        loc,
        rewriter.getStringAttr(
            "Partial Reduction Tensor Shape: [%lld, %lld]\n"),
        ArrayRef<Value>{rewriter.create<arith::ConstantIndexOp>(
                            loc, partialRegTensorShape[0]),
                        rewriter.create<arith::ConstantIndexOp>(
                            loc, partialRegTensorShape[1])});*/
    ArrayAttr tidSubTileSliceView = createInput2DView(
        loc, rewriter, op.getTidSubTileSliceView(),
        axis); // vezanoj za thread ID (tid) koji se koristi za identifikaciju
               // specifičnih niti tokom izvršavanja.
    ArrayRef<int64_t> partialReductionuctionLower2DShape =
        getLowerShape(tidSubTileSliceView);
    /*for (size_t i = 0; i < partialReductionuctionLower2DShape.size(); ++i) {
      rewriter.create<gpu::PrintfOp>(
          loc,
          rewriter.getStringAttr("Value at tidSubTileSliceView[%lld]:
    %lld\n"), ArrayRef<Value>{rewriter.create<arith::ConstantIndexOp>(loc, i),
                          rewriter.create<arith::ConstantIndexOp>(
                              loc, partialReductionuctionLower2DShape[i])});
    }*/
    partialRegTensorShape[rDim] = partialReductionuctionLower2DShape[rDim];
    ArrayAttr toFlatLDSView =
        create2DToFlatLDSView(loc, rewriter, partialRegTensorShape[nrDim],
                              partialRegTensorShape[rDim]);
    /*rewriter.create<gpu::PrintfOp>(
        loc,
        rewriter.getStringAttr(
            "Creating flat LDS view with dimensions: [%lld, %lld]\n"),
        ValueRange{rewriter.create<arith::ConstantIndexOp>(
                       loc, partialRegTensorShape[nrDim]),
                   rewriter.create<arith::ConstantIndexOp>(
                       loc, partialRegTensorShape[rDim])});*/

    storePartialReductionstoLDS(op, rewriter, loc, partialReductionBuffer,
                                workspaceLDSBuffer, inputBlockSubTile2dView,
                                inputThreadSubTile2dView, tidSubTileSliceView,
                                toFlatLDSView, blockSize);

    rewriter.create<LDSBarrierOp>(loc);

    // Following RAII scope will create reduction loops.
    {
      int64_t nonReductionDimSizeProduct = partialRegTensorShape[nrDim];
      /*Value nonReductionDimSizeProductValue =
          rewriter.create<arith::ConstantIntOp>(loc, nonReductionDimSizeProduct,
                                                64);
      rewriter.create<gpu::PrintfOp>(
          loc, rewriter.getStringAttr("nonReductionDimSizeProduct = %d\n"),
          ValueRange{nonReductionDimSizeProductValue});*/

      if (blockSize <= nonReductionDimSizeProduct) {
      // llvm::errs() << "Usao sam!!" << "\n";
      //  This means there aren't enough threads to do a parallel reduction
      //  each individual thread could do its own reduction.
      ArrayAttr threadsToTensorTrs = createThreadViewForNRLargerThanThreads(
          loc, partialRegTensorShape, blockSize, rDim, rewriter);
      ArrayAttr threadToLDSViewTrs =
          createLDSWorkspaceView(loc, rewriter, threadsToTensorTrs, rDim);
      ArrayAttr threadsToLDSViewReducedTrs = createLDSWorkspaceView(
          loc, rewriter, threadsToTensorTrs, rDim, /*makeRDimZero-*/ true);
      ArrayRef<int64_t> threadViewShape =
          cast<TransformMapAttr>(threadToLDSViewTrs[0]).getUpperBounds();

      constexpr size_t nrIterDim = 1;
      constexpr size_t rIterDim = 2;

      // Note: This currently creates a bunch of dead IR because
      // vectorization needs access to a `Value` in order to account for
      // scalarized buffers.
      Value threadToLDSViewed =
          transform(rewriter, workspaceLDSBuffer, threadToLDSViewTrs);
      {
        // Glavna logika redukcije na 256 vrednosti
        SmallVector<Value, 4> inits{tid, zeroConstantOp, zeroConstantOp};
        SmallVector<int64_t> bounds{1, 1, threadViewShape[rIterDim]};
        SmallVector<int64_t> strides{1, 1, 1}; // Ako je redukcija već urađena

        TransformingForOp reductionLoop = rewriter.create<TransformingForOp>(
            loc, ArrayRef<ValueRange>{inits, inits, inits},
            ArrayRef<Attribute>{threadToLDSViewTrs, rewriter.getArrayAttr({}),
                                threadsToLDSViewReducedTrs},
            ArrayRef<int64_t>(bounds), ArrayRef<int64_t>(strides),
            /*forceUnroll=*/true,
            /*useIndexDiffs=*/true);
        {
          PatternRewriter::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(reductionLoop.getBody());
          Block::BlockArgListType LDSLoadCoords =
              reductionLoop.getLowerCoords(/*domain=*/0);
          // There are two vectorization scenarios :
          // 1) rIterVectorLen > 1 &&  nrIterVectorLen == 1
          //    Here we will have a load vector and accReg that is a scalar
          //    The code in createReducingOp will vector reduce it before
          //    doing a reducing store to accReg
          // 2) nrIterVectorLen > 1 && rIterVectorLen == 1
          //    Here we will have a load vector and accReg that is also a
          //    vector The code in createReducingOp will do vector
          //    elementwise op and store the resulting vector to accReg.
          // NOTE: currently, LDS is viewed as [nrDim x rDim] therefore
          // only scenario 1) is exercised. However, we'd like to keep
          // this code compatible with both approaches for future changes.
          Value loadVal = rewriter.create<InBoundsLoadOp>(
              loc, elemType, workspaceLDSBuffer, LDSLoadCoords);
            /*rewriter.create<gpu::PrintfOp>(
                loc, rewriter.getStringAttr("block: %d "
                                   "tid: %d Loaded value: %f\n"),
                ArrayRef<Value>{blockId, tid, loadVal});*/
          Value rIterArg = reductionLoop.getLowerCoords(/*domain=*/1)[rIterDim];
          /*rewriter.create<gpu::PrintfOp>(
              loc, rewriter.getStringAttr("Current reduction iteration: %d\n"),
              ArrayRef<Value>{rIterArg});*/

          Value isTidZero = rewriter.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::eq, tid,
              rewriter.create<arith::ConstantIndexOp>(loc, 0));
          scf::IfOp ifb = rewriter.create<scf::IfOp>(loc, isTidZero,
                                                     /*withElseRegion=*/false);
          {
            OpBuilder thenb = ifb.getThenBodyBuilder();
            /*rewriter.create<gpu::PrintfOp>(
                loc, rewriter.getStringAttr("block: %d "
                                   "tid: %d loadVal: %f\n"),
                ValueRange{blockId, tid, loadVal});*/
            thenb.create<InBoundsStoreOp>(
                loc, loadVal, workspaceLDSBuffer,
                reductionLoop.getLowerCoords(/*domain=*/0)); //2
          }
        }
        }
        ArrayAttr reducedldsViewArrayAttr = createLDSWorkspaceView(
            loc, rewriter, inputViewArrayAttr, axis, /*makeRDimZero-*/ true,
            partialRegTensorShape[rDim]);
        rewriter.create<LDSBarrierOp>(loc);
        rewriter.create<ThreadwiseReadIntoOp>(
            loc, workspaceLDSBuffer, outputReg, reducedldsViewArrayAttr,
            /*extraIndices=*/ValueRange{tid}, true, false);
        if (ArrayAttr outputViewArrayAttr = op.getExtraOutViewAttr()) {
          ArrayAttr reducedldsViewArrayAttr2 = createLDSWorkspaceView(
              loc, rewriter, outputViewArrayAttr, axis,
              /*makeRDimZero-*/ true, partialRegTensorShape[rDim]);
          rewriter.create<ThreadwiseReadIntoOp>(
              loc, workspaceLDSBuffer, op.getExtraOut(),
              reducedldsViewArrayAttr2,
              /*extraIndices=*/ValueRange{tid}, true, false);
        }
      } else {
        // This means there are more threads than elements to be reduced.
        ArrayAttr threadToTensorViewTrs =
            createThreadViewforNRSmallerThanThreads(loc, partialRegTensorShape,
                                                    blockSize, rDim, rewriter);
        ArrayAttr threadToLDSViewTrs =
            createLDSWorkspaceView(loc, rewriter, threadToTensorViewTrs, rDim);
        ArrayRef<int64_t> threadViewShape =
            cast<TransformMapAttr>(threadToLDSViewTrs[0]).getUpperBounds();

        constexpr size_t nrIterDim = 1;
        constexpr size_t rIterDim = 2;

        // Note: This currently creates a bunch of dead IR because
        // vectorization needs access to a `Value` in order to account for
        // scalarized buffers.
        Value threadToLDSViewed =
            transform(rewriter, workspaceLDSBuffer, threadToLDSViewTrs);
        Value nrDimSizeProductConst = rewriter.create<arith::ConstantIndexOp>(
            loc, nonReductionDimSizeProduct);
        Value rtid =
            rewriter.create<arith::DivSIOp>(loc, tid, nrDimSizeProductConst);
        Value nrtid =
            rewriter.create<arith::RemSIOp>(loc, tid, nrDimSizeProductConst);

        // We need to do the threadwise reduction
        // here only if rIterDim is meaninfully iterated
        // otherwise this step can be skipped.*/
      /*if (threadViewShape[rIterDim] > 1) {
        llvm::errs() << "Usao sam!!" << "\n";
        Type accRegType = MemRefType::get({1}, elemType, AffineMap{},
                                          privateMemoryAddressSpace);
        Value accReg = rewriter.create<GpuAllocOp>(loc, accRegType);
        {
          SmallVector<Value, 4> inits{nrtid, rtid, zeroConstantOp};
          SmallVector<int64_t> bounds{1, 1, threadViewShape[rIterDim]};
          SmallVector<int64_t> strides{1, 1, 1};

          //Value initVal = getReductionInitValue(op, rewriter);
          //rewriter.create<FillOp>(loc, accReg, initVal);

          TransformingForOp reductionLoop =
              rewriter.create<TransformingForOp>(
                  loc, ArrayRef<ValueRange>(inits),
                  ArrayRef<Attribute>{threadToLDSViewTrs},
                  ArrayRef<int64_t>(bounds), ArrayRef<int64_t>(strides),
                  true, true);
          {
            PatternRewriter::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(reductionLoop.getBody());
            Block::BlockArgListType LDSLoadCoords =
                reductionLoop.getLowerCoords(0);
            Value loadVal = rewriter.create<InBoundsLoadOp>(
                loc, elemType, workspaceLDSBuffer, LDSLoadCoords);
            rewriter.create<gpu::PrintfOp>(loc,
                                           "Thread ID (rtid): %ld, "
                                           "Loaded Value: %f\n",
                                           ValueRange{rtid, loadVal});
            rewriter.create<InBoundsStoreOp>(loc, loadVal, workspaceLDSBuffer,
                                             zeroConstantOp);
          }
          rewriter.create<LDSBarrierOp>(loc);
        }
      }*/

      {
        // Glavna logika redukcije na 256 vrednosti
        SmallVector<Value, 4> inits{nrtid, rtid, zeroConstantOp};
        SmallVector<int64_t> bounds{1, 1, threadViewShape[rIterDim]};
        SmallVector<int64_t> strides{1, 1, 1}; // Ako je redukcija već urađena

        TransformingForOp reductionLoop = rewriter.create<TransformingForOp>(
            loc, ArrayRef<ValueRange>{inits, inits, inits},
            ArrayRef<Attribute>{threadToLDSViewTrs, rewriter.getArrayAttr({}),
                                threadsToLDSViewReducedTrs},
            ArrayRef<int64_t>(bounds), ArrayRef<int64_t>(strides),
            true, true);
        {
          PatternRewriter::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(reductionLoop.getBody());
          Block::BlockArgListType LDSLoadCoords =
              reductionLoop.getLowerCoords(0);
          // There are two vectorization scenarios :
          // 1) rIterVectorLen > 1 &&  nrIterVectorLen == 1
          //    Here we will have a load vector and accReg that is a scalar
          //    The code in createReducingOp will vector reduce it before
          //    doing a reducing store to accReg
          // 2) nrIterVectorLen > 1 && rIterVectorLen == 1
          //    Here we will have a load vector and accReg that is also a
          //    vector The code in createReducingOp will do vector
          //    elementwise op and store the resulting vector to accReg.
          // NOTE: currently, LDS is viewed as [nrDim x rDim] therefore
          // only scenario 1) is exercised. However, we'd like to keep
          // this code compatible with both approaches for future changes.
          Value loadVal = rewriter.create<InBoundsLoadOp>(
              loc, elemType, workspaceLDSBuffer, LDSLoadCoords);
          rewriter.create<gpu::PrintfOp>(
              loc,
              rewriter.getStringAttr(
                  "Rtid: %ld, Loaded value from workspaceLDSBuffer: %f\n"),
              ValueRange{rtid, loadVal});
          Value rIterArg =
              reductionLoop.getLowerCoords(1)[rIterDim];
          rewriter.create<gpu::PrintfOp>(
              loc,
              rewriter.getStringAttr("Current reduction iteration: %d\n"),
              ArrayRef<Value>{rIterArg});

          Value isLastIter = rewriter.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::eq, rIterArg,
              rewriter.create<arith::ConstantIndexOp>(
                  loc, threadViewShape[2] - 1));
          scf::IfOp ifb = rewriter.create<scf::IfOp>(
              loc, isLastIter, false);
          {
            OpBuilder thenb = ifb.getThenBodyBuilder();
            thenb.create<InBoundsStoreOp>(
                loc, loadVal, workspaceLDSBuffer,
                reductionLoop.getLowerCoords(2));
          }
        }
      }
      /*ArrayAttr reducedldsViewArrayAttr = createLDSWorkspaceView(
          loc, rewriter, inputViewArrayAttr, axis,  true,
          partialRegTensorShape[rDim]);
      rewriter.create<LDSBarrierOp>(loc);
      rewriter.create<ThreadwiseReadIntoOp>(
          loc, workspaceLDSBuffer, outputReg, reducedldsViewArrayAttr,
          ValueRange{tid}, true, false);
      if (ArrayAttr outputViewArrayAttr = op.getExtraOutViewAttr()) {
          llvm::errs() << "Usao sam!!" << "\n";
        ArrayAttr reducedldsViewArrayAttr2 = createLDSWorkspaceView(
            loc, rewriter, outputViewArrayAttr, axis,
             true, partialRegTensorShape[rDim]);
        rewriter.create<ThreadwiseReadIntoOp>(
            loc, workspaceLDSBuffer, op.getExtraOut(),
            reducedldsViewArrayAttr2,
            ValueRange{tid}, true, false);
      }
    }*/
      //}
      // else {
      //  This means there are more threads than elements to be reduced.
      /*ArrayAttr threadToTensorViewTrs =
          createThreadViewforNRSmallerThanThreads(loc, partialRegTensorShape,
                                                  blockSize, rDim, rewriter);
      ArrayAttr threadToLDSViewTrs =
          createLDSWorkspaceView(loc, rewriter, threadToTensorViewTrs, rDim);
      ArrayRef<int64_t> threadViewShape =
          cast<TransformMapAttr>(threadToLDSViewTrs[0]).getUpperBounds();
      SmallVector<Value, 4> dimValues;
      for (int64_t dim : threadViewShape) {
        dimValues.push_back(
            rewriter.create<arith::ConstantIndexOp>(loc, dim));
      }

      // Kreiranje format stringa za PrintfOp
      std::string formatStr = "Thread view shape: [";
      for (size_t i = 0; i < threadViewShape.size(); ++i) {
        formatStr += "%d ";
      }
      formatStr += "]\n";

      // Ispisivanje threadViewShape koristeći gpu::PrintfOp
      rewriter.create<gpu::PrintfOp>(loc, rewriter.getStringAttr(formatStr),
                                     dimValues);
      constexpr size_t rTidDim = 1;
      constexpr size_t rIterDim = 2;

      Value threadToLDSViewed =
          transform(rewriter, workspaceLDSBuffer, threadToLDSViewTrs);
      VectorizationResult rIterVectorRes =
          getMaxVectorization(threadToLDSViewed, rIterDim);
      int64_t rIterVectorLen = rIterVectorRes.max;
      Value nrDimSizeProductConst = rewriter.create<arith::ConstantIndexOp>(
          loc, nonReductionDimSizeProduct);
      Value rtid =
          rewriter.create<arith::DivSIOp>(loc, tid, nrDimSizeProductConst);
      Value nrtid =
          rewriter.create<arith::RemSIOp>(loc, tid, nrDimSizeProductConst);

      // We need to do the threadwise reduction
      // here only if rIterDim is meaninfully iterated
      // otherwise this step can be skipped.
      if (threadViewShape[rIterDim] > 1) {
        // This is where thread_wise reduction result is stored.
        Type loadTypeInputReg = vectorTypeOrSelf(elemType, rIterVectorLen);
        Type accRegType = MemRefType::get({1}, elemType, AffineMap{},
                                          privateMemoryAddressSpace);
        Value accReg = rewriter.create<GpuAllocOp>(loc, accRegType);
        // This RAII scope would create a loop to iteratively partialy
        // reduce on a thread basis until items to reduce will match the
        // available number of threads.
        {
          SmallVector<Value, 4> inits{nrtid, rtid, zeroConstantOp};
          SmallVector<int64_t> bounds{1, 1, threadViewShape[rIterDim]};
          SmallVector<int64_t> strides{1, 1, rIterVectorLen};

          Value initVal = getReductionInitValue(op, rewriter);
          rewriter.create<FillOp>(loc, accReg, initVal);

          TransformingForOp reductionLoop =
              rewriter.create<TransformingForOp>(
                  loc, ArrayRef<ValueRange>(inits),
                  ArrayRef<Attribute>{threadToLDSViewTrs},
                  ArrayRef<int64_t>(bounds), ArrayRef<int64_t>(strides),*/
      /*forceUnroll=*/ // true, /*useIndexDiffs=*/true);
      /*{
        PatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(reductionLoop.getBody());
        Block::BlockArgListType LDSLoadCoords =
            reductionLoop.getLowerCoords(*/
      /*domain=*/ // 0);
      /*Value loadVal = rewriter.create<InBoundsLoadOp>(
          loc, loadTypeInputReg, workspaceLDSBuffer, LDSLoadCoords);
      Value loadAcc = rewriter.create<InBoundsLoadOp>(
          loc, elemType, accReg, zeroConstantOp);
      Value reduced = createReducingOp(op, loadVal, loadAcc, rewriter);
      // Print all relevant values in one line
      rewriter.create<gpu::PrintfOp>(
          loc,
          "Thread ID (rtid): %ld, Accumulator before reduction: %f, "
          "Reduced Value: %f\n",
          ValueRange{rtid, loadAcc, reduced});
      rewriter.create<InBoundsStoreOp>(loc, reduced, accReg,
                                       zeroConstantOp);
    }
  }

  // This RAII scope would store the partial reductions to
  // LDS
  {
    SmallVector<Value, 4> inits{nrtid, rtid, zeroConstantOp};
    SmallVector<int64_t> bounds{1, 1, 1};
    SmallVector<int64_t> strides{1, 1, 1};

    TransformingForOp reductionLoop =
        rewriter.create<TransformingForOp>(
            loc, ArrayRef<ValueRange>(inits),
            ArrayRef<Attribute>{threadToLDSViewTrs},
            ArrayRef<int64_t>(bounds), ArrayRef<int64_t>(strides),*/
      /*forceUnroll=*/ // true, /*useIndexDiffs=*/true);
      /*{
        PatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(reductionLoop.getBody());
        Block::BlockArgListType LDSStoreCoords =
            reductionLoop.getLowerCoords(*/
      /*domain=*/ // 0);
      /*Value loadVal = rewriter.create<InBoundsLoadOp>(
           loc, elemType, accReg, zeroConstantOp);
       rewriter.create<InBoundsStoreOp>(loc, loadVal, workspaceLDSBuffer,
                                        LDSStoreCoords);
     }
     rewriter.create<LDSBarrierOp>(loc);
   }
  }

  // This RAII scope would do the following :
  // LDS[rtid] = reduce(LDS[rtid], LDS[rtid + offset])
  // where offset is a power of 2.
  // Initial it starts with power = ceil(|rtid|, power of 2) / 2
  // Then keep on reducing the power.
  {
   int64_t ceilPowerOf2 =
       llvm::PowerOf2Ceil(threadViewShape[rTidDim]) / 2;
   auto ceilPowerOf2MLIR =
       rewriter.create<arith::ConstantIndexOp>(loc, ceilPowerOf2);
   rewriter.create<mlir::gpu::PrintfOp>(loc, "ceilPowerOf2: %d\n",
                                        ValueRange{ceilPowerOf2MLIR});
   int64_t maxActiveReductionThreads = threadViewShape[rTidDim];
   auto cmaxActiveReductionThreadsMLIR =
       rewriter.create<arith::ConstantIndexOp>(
           loc, maxActiveReductionThreads);
   rewriter.create<mlir::gpu::PrintfOp>(
       loc, "maxActiveReductionThreads: %d\n",
       ValueRange{cmaxActiveReductionThreadsMLIR});

   for (int64_t offset = ceilPowerOf2; offset >= 1;
        offset = offset >> 1) {
     Value offsetVal =
         rewriter.create<arith::ConstantIndexOp>(loc, offset);
     Value rtidPlusOffsetVal =
         rewriter.create<arith::AddIOp>(loc, rtid, offsetVal);
     Value maxActiveReductionThreadsVal =
         rewriter.create<arith::ConstantIndexOp>(
             loc, maxActiveReductionThreads);
     rewriter.create<mlir::gpu::PrintfOp>(
         loc,
         "offsetVal:  %d\n, rtidPlusOffsetVal: %d\n, "
         "maxActiveReductionThreadsVal: %d\n",
         ValueRange{offsetVal, rtidPlusOffsetVal,
                    maxActiveReductionThreadsVal});
     maxActiveReductionThreads =
         llvm::PowerOf2Ceil(maxActiveReductionThreads) >> 1;
     Value isValid = rewriter.create<arith::CmpIOp>(
         loc, arith::CmpIPredicate::slt, rtidPlusOffsetVal,
         maxActiveReductionThreadsVal);
     scf::IfOp ifb = rewriter.create<scf::IfOp>(
         loc, isValid, */
      /*withElseRegion=*/ // false);
      /*{
        OpBuilder thenb = ifb.getThenBodyBuilder();
        SmallVector<Value, 4> firstInits{nrtid, rtid, zeroConstantOp};
        SmallVector<Value, 4> secondInits{nrtid, rtidPlusOffsetVal,
                                          zeroConstantOp};
        SmallVector<int64_t> bounds{1, 1, 1};
        SmallVector<int64_t> strides{1, 1, 1};

        TransformingForOp reductionLoop = thenb.create<TransformingForOp>(
            loc, ArrayRef<ValueRange>{firstInits, secondInits},
            ArrayRef<Attribute>{threadToLDSViewTrs, threadToLDSViewTrs},
            ArrayRef<int64_t>(bounds), ArrayRef<int64_t>(strides),*/
      /*forceUnroll=*/ // true, /*useIndexDiffs=*/true);
      /*{
        PatternRewriter::InsertionGuard guard(thenb);
        thenb.setInsertionPointToStart(reductionLoop.getBody());
        Block::BlockArgListType firstLDSLoadCoords =
            reductionLoop.getLowerCoords(0);
        Value firstLoadVal = thenb.create<InBoundsLoadOp>(
            loc, elemType, workspaceLDSBuffer, firstLDSLoadCoords);
        Block::BlockArgListType secondLDSLoadCoords =
            reductionLoop.getLowerCoords(1);
        Value secondLoadVal = thenb.create<InBoundsLoadOp>(
            loc, elemType, workspaceLDSBuffer, secondLDSLoadCoords);
        Value reduced =
            createReducingOp(op, firstLoadVal, secondLoadVal, thenb);
        thenb.create<mlir::gpu::PrintfOp>(
            loc,
            "Thread ID: %d\n, rtid: %d\n,  First Load Value: %f\n, "
            "Second Load Value: %f\n, Reduced Value: %f\n",
            ValueRange{tid, rtid, firstLoadVal, secondLoadVal,
                       reduced});
        thenb.create<InBoundsStoreOp>(loc, reduced, workspaceLDSBuffer,
                                      firstLDSLoadCoords);
      }
    }
    rewriter.create<LDSBarrierOp>(loc);
  }
  ArrayAttr reducedldsViewArrayAttr = createLDSWorkspaceView(
      loc, rewriter, inputViewArrayAttr, axis, */
      /*makeRDimZero-*/ // true,
                        /**partialRegTensorShape[rDim]);
                     rewriter.create<ThreadwiseReadIntoOp>(
                         loc, workspaceLDSBuffer, outputReg, reducedldsViewArrayAttr,*/
      /*extraIndices=*/ // ValueRange{tid}, true, false);
      /*if (ArrayAttr outputViewArrayAttr = op.getExtraOutViewAttr()) {
        ArrayAttr reducedldsViewArrayAttr2 = createLDSWorkspaceView(
            loc, rewriter, outputViewArrayAttr, axis,*/
      /*makeRDimZero-*/ // true, partialRegTensorShape[rDim]);
                        /*rewriter.create<ThreadwiseReadIntoOp>(
                             loc, workspaceLDSBuffer, op.getExtraOut(),
                             reducedldsViewArrayAttr2,*/
      /*extraIndices=*/ // ValueRange{tid}, true, false);
      /*}
    }
  }*/
      rewriter.eraseOp(op);
      return success();
    }
  }
};

void RockLowerBlockwiseGemmToThreadwisePass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  {
    ConversionTarget writeAllTarget(*ctx);
    
    writeAllTarget.addIllegalOp<BlockwiseBroadcastReduceOp, BlockwiseFillOp>();
    writeAllTarget.addLegalDialect<arith::ArithDialect, rock::RockDialect,
                                   memref::MemRefDialect, scf::SCFDialect,
                                   vector::VectorDialect, AffineDialect,
                                   ROCDL::ROCDLDialect, amdgpu::AMDGPUDialect>();
    writeAllTarget.addLegalOp<gpu::PrintfOp>();
    RewritePatternSet writeAllPatterns(ctx);
    writeAllPatterns
        .add<BlockwiseReduceRewritePattern, BlockwiseFillRewritePattern>(ctx);
    if (failed(applyPartialConversion(getOperation(), writeAllTarget,
                                      std::move(writeAllPatterns))))
      signalPassFailure();
  }

  ConversionTarget target(*ctx);
  target.addIllegalOp<FillOp, BlockwiseGemmOp, BlockwiseGemmAccelOp>();
  target.addLegalDialect<arith::ArithDialect, rock::RockDialect,
                         affine::AffineDialect, vector::VectorDialect,
                         memref::MemRefDialect, ROCDL::ROCDLDialect,
                         amdgpu::AMDGPUDialect>();
  target.addLegalOp<gpu::PrintfOp>();

  RewritePatternSet patterns(ctx);
  patterns.add<FillRewritePattern, BlockwiseGemmRewritePattern,
               BlockwiseGemmAccelRewritePattern>(ctx);
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
} // end anonymous namespace
