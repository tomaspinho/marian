#include "translation_task.h"

#include <string>

#include "search.h"
#include "output_collector.h"
#include "printer.h"
#include "history.h"

using namespace std;

namespace amunmt {

void TranslationTask::RunMaxiBatchAndOutput(God &god, SentencesPtr maxiBatch, size_t miniSize, int miniWords)
{
  maxiBatch->SortByLength();
  while (maxiBatch->size()) {
    SentencesPtr miniBatch = maxiBatch->NextMiniBatch(miniSize, miniWords);
    //cerr << "miniBatch=" << miniBatch->size() << " maxiBatch=" << maxiBatch->size() << endl;

    god.GetThreadPool().enqueue(
        [&,miniBatch]{ return RunAndOutput(god, miniBatch); }
        );
  }

}

void TranslationTask::RunAndOutput(const God &god, SentencesPtr sentences) {
  Search& search = god.GetSearch();

  OutputCollector &outputCollector = god.GetOutputCollector();

  std::shared_ptr<Histories> histories = Run(god, sentences);

  for (size_t i = 0; i < histories->size(); ++i) {
    const History &history = *histories->at(i);
    size_t lineNum = history.GetLineNum();

    std::stringstream strm;
    search.Printer(god, history, strm);

    outputCollector.Write(lineNum, strm.str());
  }
}

std::shared_ptr<Histories> TranslationTask::Run(const God &god, SentencesPtr sentences) {
  try {
    Search& search = god.GetSearch();
    auto histories = search.Translate(*sentences);

    return histories;
  }
#ifdef CUDA
  catch(thrust::system_error &e)
  {
    std::cerr << "CUDA error during some_function: " << e.what() << std::endl;
    abort();
  }
#endif
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

}

