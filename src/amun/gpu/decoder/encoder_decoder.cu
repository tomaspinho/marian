// -*- mode: c++; tab-width: 2; indent-tabs-mode: nil -*-
#include <iostream>

#include "common/god.h"
#include "common/sentences.h"
#include "common/search.h"
#include "common/histories.h"

#include "encoder_decoder.h"
#include "gpu/mblas/matrix_functions.h"
#include "gpu/decoder/enc_out_gpu.h"
#include "gpu/dl4mt/dl4mt.h"
#include "gpu/decoder/encoder_decoder_state.h"
#include "gpu/decoder/best_hyps.h"
#include "gpu/decoder/beam_size_gpu.h"

using namespace std;

namespace amunmt {
namespace GPU {

///////////////////////////////////////////////////////////////////////////////
EncoderDecoder::EncoderDecoder(
        const God &god,
        const std::string& name,
        const YAML::Node& config,
        size_t tab,
        const Weights& model,
        const Search &search)
: Scorer(god, name, config, tab, search),
  model_(model),
  encoder_(new Encoder(model_)),
  decoder_(new Decoder(god, model_)),
  encDecBuffer_(3)

{
  std::thread *thread = new std::thread( [&]{ DecodeAsync(god); });
  decThread_.reset(thread);

}

EncoderDecoder::~EncoderDecoder()
{
  decThread_->join();
}

State* EncoderDecoder::NewState() const {
  return new EDState();
}

void EncoderDecoder::Encode(const SentencesPtr source) {
  BEGIN_TIMER("Encode");

  EncOutPtr encOut(new EncOutGPU(source));

  if (source->size()) {
    encoder_->Encode(god_, *source, tab_, encOut);

    mblas::Matrix &bufStates = encOut->GetStates<mblas::Matrix>();
    mblas::Matrix &bufEmbeddings = encOut->GetEmbeddings<mblas::Matrix>();

    const mblas::Matrix &sourceContext = encOut->GetSourceContext<mblas::Matrix>();
    const mblas::IMatrix &sourceLengths = encOut->GetSentenceLengths<mblas::IMatrix>();
    size_t batchSize = encOut->GetSentences().size();

    mblas::Matrix &SCU = encOut->GetSCU<mblas::Matrix>();

    BeginSentenceState(bufStates,
                      bufEmbeddings,
                      SCU,
                      sourceContext,
                      sourceLengths,
                      batchSize);

  }

  encDecBuffer_.Add(encOut);
  //cerr << "Encode encOut->sourceContext_=" << encOut->sourceContext_.Debug(0) << endl;

  PAUSE_TIMER("Encode");
}

void EncoderDecoder::BeginSentenceState(mblas::Matrix &states,
                                        mblas::Matrix &embeddings,
                                        mblas::Matrix &SCU,
                                        const mblas::Matrix &sourceContext,
                                        const mblas::IMatrix &sourceLengths,
                                        size_t batchSize) const
{
  decoder_->EmptyState(states,
                      SCU,
                      sourceContext,
                      sourceLengths,
                      batchSize);

  decoder_->EmptyEmbedding(embeddings, batchSize);
}

void EncoderDecoder::DecodeAsync(const God &god)
{
  //cerr << "BeginSentenceState encOut->sourceContext_=" << encOut->sourceContext_.Debug(0) << endl;
  try {
    DecodeAsyncInternal(god);
  }
  catch(thrust::system_error &e)
  {
    std::cerr << "CUDA error during some_function: " << e.what() << std::endl;
    abort();
  }
  catch(std::bad_alloc &e)
  {
    std::cerr << "Bad memory allocation during some_function: " << e.what() << std::endl;
    abort();
  }
  catch(std::runtime_error &e)
  {
    std::cerr << "Runtime error during some_function: " << e.what() << std::endl;
    abort();
  }
  catch(...)
  {
    std::cerr << "Some other kind of error during some_function" << std::endl;
    abort();
  }
}

void EncoderDecoder::DecodeAsyncInternal(const God &god)
{
  boost::timer::cpu_timer timer;

  uint maxBeamSize = god.Get<uint>("beam-size");

  EDState state;

  uint remaining;
  Hypotheses prevHyps;
  Histories histories(new BeamSizeGPU(), search_.NormalizeScore());
  mblas::Matrix SCU;
  size_t decoderStep;

  while (true) {
    if (histories.size() == 0) {
      // clean up previous
      CleanUpAfterSentence();

      LOG(progress)->info("Decoding took {}", timer.format(3, "%ws"));

      // read in next batch
      EncOutPtr encOut = encDecBuffer_.Get();
      assert(encOut);

      const Sentences &sentences = encOut->GetSentences();
      if (sentences.size() == 0) {
        break;
      }

      timer.start();

      //cerr << "sentences=" << sentences.size() << " " << sentences.GetMaxLength() << endl;

      // simulate completed vector
      remaining = god.Get<uint>("mini-batch");
      std::vector<uint> completed2(remaining);
      for (uint i = 0; i < remaining; ++i) {
        completed2[i] = i;
      }

      // init states & histories/beams
      const mblas::Matrix &bufStates = encOut->GetStates<mblas::Matrix>();
      const mblas::Matrix &bufEmbeddings = encOut->GetEmbeddings<mblas::Matrix>();
      const mblas::Matrix &bufSCU = encOut->GetSCU<mblas::Matrix>();

      mblas::Matrix &states = state.GetStates();
      mblas::Matrix &embeddings = state.GetEmbeddings();
      states.Copy(bufStates);
      embeddings.Copy(bufEmbeddings);
      SCU.Copy(bufSCU);

      /*
      cerr << "1state=" << state.Debug(1) << endl;
      cerr << "1SCU=" << SCU.Debug(1) << endl;
      */

      histories.Init(maxBeamSize, encOut);
      prevHyps = histories.GetFirstHyps();

      decoderStep = 0;
    }

    // decode
    boost::timer::cpu_timer timerStep;

    mblas::Matrix nextStateMatrix;
    mblas::Matrix attention;
    mblas::Matrix probs;

   /*
    cerr << "2state=" << state.Debug(1) << endl;
    cerr << "2SCU=" << SCU.Debug(1) << endl;
    cerr << "2nextStateMatrix=" << nextStateMatrix.Debug(1) << endl;
    cerr << "2probs_=" << probs.Debug(1) << endl;
    cerr << "2attention_=" << attention.Debug(1) << endl;
    */

    const BeamSizeGPU &bsGPU = static_cast<const BeamSizeGPU&>(histories.GetBeamSizes());

    BEGIN_TIMER("Decode");
    decoder_->Decode(nextStateMatrix,
                    probs,
                    attention,
                    state.GetStates(),
                    state.GetEmbeddings(),
                    SCU,
                    bsGPU);
    PAUSE_TIMER("Decode");

    /*
    cerr << "3state=" << state.Debug(1) << endl;
    cerr << "3SCU=" << SCU.Debug(1) << endl;
    cerr << "3nextStateMatrix=" << nextStateMatrix.Debug(1) << endl;
    cerr << "3probs_=" << probs.Debug(1) << endl;
    cerr << "3attention_=" << attention.Debug(1) << endl;
    */
    // beams
    histories.SetNewBeamSize(search_.MaxBeamSize());

    Beams beams;
    search_.BestHyps()->CalcBeam(prevHyps, probs, attention, *this, search_.FilterIndices(), beams, histories.GetBeamSizes());

    /*
    cerr << "4state=" << state.Debug(1) << endl;
    cerr << "4SCU=" << SCU.Debug(1) << endl;
    cerr << "4nextStateMatrix=" << nextStateMatrix.Debug(1) << endl;
    cerr << "4probs_=" << probs.Debug(1) << endl;
    cerr << "4attention_=" << attention.Debug(1) << endl;
    */
    std::pair<Hypotheses, std::vector<uint> > histOut = histories.AddAndOutput(god, beams);
    Hypotheses &survivors = histOut.first;
    const std::vector<uint> &completed = histOut.second;

    /*
    cerr << "5state=" << state.Debug(1) << endl;
    cerr << "5SCU=" << SCU.Debug(1) << endl;
    cerr << "5nextStateMatrix=" << nextStateMatrix.Debug(1) << endl;
    cerr << "5probs_=" << probs.Debug(1) << endl;
    cerr << "5attention_=" << attention.Debug(1) << endl;
    */
    AssembleBeamState(nextStateMatrix, survivors, state);
    /*
    cerr << "6state=" << state.Debug(1) << endl;
    cerr << "6SCU=" << SCU.Debug(1) << endl;
    cerr << "6nextStateMatrix=" << nextStateMatrix.Debug(1) << endl;
    cerr << "6probs_=" << probs.Debug(1) << endl;
    cerr << "6attention_=" << attention.Debug(1) << endl;
    */

    size_t numCompleted = completed.size();
    std::vector<EncOut::SentenceElement> newSentences;

    if (numCompleted) {
      encDecBuffer_.Get(numCompleted, newSentences);
    }

    BeamSizeGPU &bsGPU2 = static_cast<BeamSizeGPU&>(histories.GetBeamSizes());

    ShrinkBatch(completed,
                histories.GetBeamSizes(),
                bsGPU2.GetSourceContext(),
                bsGPU2.GetSentenceLengths(),
                SCU);

    AddToBatch(newSentences,
              histories.GetBeamSizes(),
              bsGPU2.GetSourceContext(),
              bsGPU2.GetSentenceLengths(),
              SCU,
              state.GetStates(),
              state.GetEmbeddings());

    prevHyps.swap(survivors);
    ++decoderStep;
    remaining -= completed.size();


    LOG(progress)->info("Step took {}, survivors={}, completed={}, remaining={}",
                        timerStep.format(3, "%ws"),
                        survivors.size(),
                        completed.size(),
                        remaining);

    /*
    cerr << "3 nextState=" << nextStateMatrix.Debug(1) << endl;
    cerr << "3 probs=" << probs.Debug(1) << endl;
    cerr << "3 attention=" << attention.Debug(2) << endl;

    cerr << "beamSizes=" << Debug(beamSizes, 2) << endl;
    cerr << "survivors=" << survivors.size() << endl;
    cerr << "beams=" << beams.size() << endl;
    cerr << "beamSizes5=" << histories.GetBeamSizes().Debug(2) << endl;
    cerr << "histories=" << histories.size() << endl;
    cerr << "3 state=" << state.Debug(1) << endl;
    cerr << "3SCU=" << SCU.Debug(1) << endl;
    cerr << "completed=" << Debug(completed, 2) << endl;
    cerr << "newSentences=" << newSentences.size() << endl;
    */
    cerr << endl;
  }
}


void EncoderDecoder::AssembleBeamState(const mblas::Matrix &nextStateMatrix,
                                const Hypotheses& hypos,
                                EDState& out) const
{
  if (hypos.size() == 0) {
    return;
  }

  std::vector<size_t> beamWords;
  std::vector<uint> beamStateIds;
  for (const HypothesisPtr &h : hypos) {
     beamWords.push_back(h->GetWord());
     beamStateIds.push_back(h->GetPrevStateIndex());
  }
  //cerr << "beamWords=" << Debug(beamWords, 2) << endl;
  //cerr << "beamStateIds=" << Debug(beamStateIds, 2) << endl;

  DeviceVector<uint> indices(beamStateIds.size());
  //HostVector<uint> tmp = beamStateIds;

  //cerr << "3 beamStateIds=" << Debug(beamStateIds, 2) << endl;

  mblas::copy(beamStateIds.data(),
      beamStateIds.size(),
      thrust::raw_pointer_cast(indices.data()),
      cudaMemcpyHostToDevice);

  mblas::Assemble(out.GetStates(), nextStateMatrix, indices);
  //cerr << "out.GetStates()=" << out.GetStates().Debug(0) << endl;
  //cerr << "nextStateMatrix=" << nextStateMatrix.Debug(0) << endl;

  //cerr << "beamWords=" << Debug(beamWords, 2) << endl;
  decoder_->Lookup(out.GetEmbeddings(), beamWords);
  //cerr << "edOut.GetEmbeddings()=" << edOut.GetEmbeddings().Debug(1) << endl;
}

BaseMatrix& EncoderDecoder::GetProbs() {
  assert(false);
  return *(new mblas::Matrix());
}

size_t EncoderDecoder::GetVocabSize() const {
  return decoder_->GetVocabSize();
}

void EncoderDecoder::Filter(const std::vector<size_t>& filterIds) {
  decoder_->Filter(filterIds);
}

void EncoderDecoder::ShrinkBatch(const std::vector<uint> &completed,
                                BeamSize &beamSize,
                                mblas::Matrix &sourceContext,
                                mblas::IMatrix &sentenceLengths,
                                mblas::Matrix &SCU)
{
  //cerr << "completed=" << Debug(completed, 2) << endl;
  if (completed.size() == 0) {
    return;
  }

  // shrink beam
  size_t origBeamSize = beamSize.size();
  beamSize.DeleteEmpty(completed);
  size_t newBeamSize = beamSize.size();
  //cerr << "origBeamSize=" << origBeamSize << " newBeamSize=" << newBeamSize << endl;

  // old ind -> new ind
  std::vector<uint> newIndices(newBeamSize, 99999);
  uint newInd = 0;
  uint completedInd = 0;
  for (size_t origInd = 0; origInd < origBeamSize; ++origInd) {
    if (completedInd < completed.size() && completed[completedInd] == origInd) {
      ++completedInd;
    }
    else {
      newIndices[newInd] = origInd;
      ++newInd;
    }
  }
  //cerr << "newIndices=" << Debug(newIndices, 2) << endl;

  // shrink matrices

  size_t sizeShrink = completed.size();
  DeviceVector<uint> d_newIndices(newIndices);
  ShrinkMatrix(sizeShrink, d_newIndices, 3, sourceContext);
  ShrinkMatrix(sizeShrink, d_newIndices, 3, SCU);

  ShrinkMatrix(sizeShrink, d_newIndices, 0, sentenceLengths);
}

void EncoderDecoder::AddToBatch(const std::vector<EncOut::SentenceElement> &newSentences,
                BeamSize &beamSize,
                mblas::Matrix &sourceContext,
                mblas::IMatrix &sentenceLengths,
                mblas::Matrix &SCU,
                mblas::Matrix &states,
                mblas::Matrix &embeddings)
{
  /*
  cerr << "newSentences=" << newSentences.size() << endl;
  cerr << "sourceContext=" << sourceContext.Debug(0) << endl;
  cerr << "sentenceLengths=" << sentenceLengths.Debug(0) << endl;
  cerr << "SCU=" << SCU.Debug(0) << endl;
  cerr << "1states=" << states.Debug(0) << endl;
  cerr << "1embeddings=" << embeddings.Debug(0) << endl;
  */
  size_t currBatchInd = beamSize.size();
  size_t currHypoInd = states.dim(0);

  beamSize.AddNewSentences(newSentences);

  uint numNewSentences = newSentences.size();
  EnlargeMatrix(3, numNewSentences, sourceContext);
  EnlargeMatrix(0, numNewSentences, sentenceLengths);
  EnlargeMatrix(3, numNewSentences, SCU);
  EnlargeMatrix(0, numNewSentences, states);
  EnlargeMatrix(0, numNewSentences, embeddings);

  //cerr << "2states=" << states.Debug(0) << endl;
  //cerr << "2embeddings=" << embeddings.Debug(0) << endl;

  for (size_t i = 0; i < newSentences.size(); ++i) {
    const EncOut::SentenceElement &ele = newSentences[i];
    const EncOutPtr encOut = ele.encOut;
    size_t sentenceInd = ele.sentenceInd;

    const mblas::Matrix &origSourceContext = encOut->GetSourceContext<mblas::Matrix>();
    const mblas::IMatrix &origSentenceLengths = encOut->GetSentenceLengths<mblas::IMatrix>();
    const mblas::Matrix &origSCU = encOut->GetSCU<mblas::Matrix>();
    const mblas::Matrix &origStates = encOut->GetStates<mblas::Matrix>();
    const mblas::Matrix &origEmbeddings = encOut->GetEmbeddings<mblas::Matrix>();

    /*
    cerr << "sentenceInd=" << sentenceInd << endl;
    cerr << "currBatchInd=" << currBatchInd << endl;
    cerr << "currHypoInd=" << currHypoInd << endl;
    cerr << "origSourceContext=" << origSourceContext.Debug(0) << endl;
    cerr << "origSentenceLengths=" << origSentenceLengths.Debug(0) << endl;
    cerr << "origSCU=" << origSCU.Debug(0) << endl;
    cerr << "origStates=" << origStates.Debug(0) << endl;
    cerr << "origEmbeddings=" << origEmbeddings.Debug(0) << endl;
    */

    assert(currBatchInd < sourceContext.dim(3));
    mblas::CopyDimension<float>(3, currBatchInd, sentenceInd, sourceContext, origSourceContext);

    assert(currBatchInd < sentenceLengths.dim(0));
    mblas::CopyDimension<uint>(0, currBatchInd, sentenceInd, sentenceLengths, origSentenceLengths);

    assert(currBatchInd < SCU.dim(3));
    mblas::CopyDimension<float>(3, currBatchInd, sentenceInd, SCU, origSCU);

    assert(currBatchInd < states.dim(0));
    mblas::CopyDimension<float>(0, currHypoInd, sentenceInd, states, origStates);

    assert(currBatchInd < embeddings.dim(0));
    mblas::CopyDimension<float>(0, currHypoInd, sentenceInd, embeddings, origEmbeddings);

    ++currBatchInd;
    ++currHypoInd;
  }
}

}
}

