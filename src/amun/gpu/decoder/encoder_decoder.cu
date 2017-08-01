// -*- mode: c++; tab-width: 2; indent-tabs-mode: nil -*-
#include <iostream>

#include "common/god.h"
#include "common/sentences.h"
#include "common/search.h"
#include "common/histories.h"

#include "encoder_decoder.h"
#include "gpu/mblas/matrix_functions.h"
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
  indices_(god.Get<size_t>("beam-size")),
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

  mblas::EncParamsPtr encParams(new mblas::EncParamsGPU());
  encParams->sentences = source;

  if (source->size()) {
    encoder_->Encode(*source, tab_, encParams);
  }

  encDecBuffer_.add(encParams);
  //cerr << "Encode encParams->sourceContext_=" << encParams->sourceContext_.Debug(0) << endl;

  PAUSE_TIMER("Encode");
}

void EncoderDecoder::BeginSentenceState(State& state, size_t batchSize, mblas::EncParamsPtr encParams)
{
  //cerr << "BeginSentenceState encParams->sourceContext_=" << encParams->sourceContext_.Debug(0) << endl;
  //cerr << "BeginSentenceState encParams->sentencesMask_=" << encParams->sentencesMask_.Debug(0) << endl;
  //cerr << "batchSize=" << batchSize << endl;

  EDState& edState = state.get<EDState>();

  decoder_->EmptyState(edState.GetStates(), encParams, batchSize);

  decoder_->EmptyEmbedding(edState.GetEmbeddings(), batchSize);
}

void EncoderDecoder::Decode(const State& in, State& out, const BeamSize& beamSizes) {
  BEGIN_TIMER("Decode");
  const EDState& edIn = in.get<EDState>();
  EDState& edOut = out.get<EDState>();

  const BeamSizeGPU &bs = static_cast<const BeamSizeGPU&>(beamSizes);

  decoder_->Decode(edOut.GetStates(),
                     edIn.GetStates(),
                     edIn.GetEmbeddings(),
                     bs);
  PAUSE_TIMER("Decode");
}


void EncoderDecoder::DecodeAsync(const God &god)
{
  //cerr << "BeginSentenceState encParams->sourceContext_=" << encParams->sourceContext_.Debug(0) << endl;
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
  while (true) {
    mblas::EncParamsPtr encParams = encDecBuffer_.remove();
    assert(encParams.get());
    assert(encParams->sentences.get());

    if (encParams->sentences->size() == 0) {
      return;
    }

    boost::timer::cpu_timer timer;

    // begin decoding - create 1st decode states
    State *state = NewState();
    BeginSentenceState(*state, encParams->sentences->size(), encParams);

    State *nextState = NewState();

    Histories histories(new BeamSizeGPU(encParams), search_.NormalizeScore());
    Hypotheses prevHyps = histories.GetFirstHyps();

    cerr << "beamSizes1=" << histories.GetBeamSizes().Debug(2) << endl;

    // decode
    for (size_t decoderStep = 0; decoderStep < 3 * encParams->sentences->GetMaxLength(); ++decoderStep) {
      boost::timer::cpu_timer timerStep;

      //cerr << "beamSizes2=" << beamSizes.Debug(2) << endl;
      Decode(*state, *nextState, histories.GetBeamSizes());

      cerr << "beamSizes3=" << histories.GetBeamSizes().Debug(2) << endl;
      cerr << "state=" << state->Debug(0) << endl;

      // beams
      if (decoderStep == 0) {
        histories.InitBeamSize(search_.MaxBeamSize());
      }
      //cerr << "beamSizes4=" << beamSizes.Debug(2) << endl;

      Beams beams;
      search_.BestHyps()->CalcBeam(prevHyps, *this, search_.FilterIndices(), beams, histories.GetBeamSizes());

      Hypotheses survivors = histories.AddAndOutput(god, beams);

      cerr << "beamSizes5=" << histories.GetBeamSizes().Debug(2) << endl;

      /*
      cerr << "beamSizes=" << Debug(beamSizes, 2) << endl;
      cerr << "survivors=" << survivors.size() << endl;
      cerr << "beams=" << beams.size() << endl;
      cerr << "histories=" << histories.size() << endl;
      cerr << "state=" << state->Debug(0) << endl;
      cerr << "nextState=" << nextState->Debug(0) << endl;
      */

      if (survivors.size() == 0) {
        break;
      }

      AssembleBeamState(*nextState, survivors, *state);

      //beamSizes.DeleteEmpty();
      //cerr << "beamSizes6=" << beamSizes.Debug(2) << endl;

      prevHyps.swap(survivors);

      cerr << endl;
      LOG(progress)->info("Step took {}", timerStep.format(3, "%ws"));
    } // for (size_t decoderStep = 0; decoderStep < 3 * encParams->sentences->GetMaxLength(); ++decoderStep) {

    histories.OutputRemaining(god);

    CleanUpAfterSentence();

    // output
    //Output(god, histories);

    LOG(progress)->info("Decoding took {}", timer.format(3, "%ws"));
  }
}


void EncoderDecoder::AssembleBeamState(const State& in,
                               const Hypotheses& hypos,
                               State& out) {
  std::vector<size_t> beamWords;
  std::vector<uint> beamStateIds;
  for (const HypothesisPtr &h : hypos) {
     beamWords.push_back(h->GetWord());
     beamStateIds.push_back(h->GetPrevStateIndex());
  }
  //cerr << "beamWords=" << Debug(beamWords, 2) << endl;
  //cerr << "beamStateIds=" << Debug(beamStateIds, 2) << endl;

  const EDState& edIn = in.get<EDState>();
  EDState& edOut = out.get<EDState>();
  indices_.resize(beamStateIds.size());
  HostVector<uint> tmp = beamStateIds;

  mblas::copy(thrust::raw_pointer_cast(tmp.data()),
      beamStateIds.size(),
      thrust::raw_pointer_cast(indices_.data()),
      cudaMemcpyHostToDevice);
  //cerr << "indices_=" << mblas::Debug(indices_, 2) << endl;

  mblas::Assemble(edOut.GetStates(), edIn.GetStates(), indices_);
  //cerr << "edOut.GetStates()=" << edOut.GetStates().Debug(1) << endl;

  //cerr << "beamWords=" << Debug(beamWords, 2) << endl;
  decoder_->Lookup(edOut.GetEmbeddings(), beamWords);
  //cerr << "edOut.GetEmbeddings()=" << edOut.GetEmbeddings().Debug(1) << endl;
}

void EncoderDecoder::GetAttention(mblas::Matrix& Attention) {
  decoder_->GetAttention(Attention);
}

BaseMatrix& EncoderDecoder::GetProbs() {
  return decoder_->GetProbs();
}

mblas::Matrix& EncoderDecoder::GetAttention() {
  return decoder_->GetAttention();
}

size_t EncoderDecoder::GetVocabSize() const {
  return decoder_->GetVocabSize();
}

void EncoderDecoder::Filter(const std::vector<size_t>& filterIds) {
  decoder_->Filter(filterIds);
}


}
}

