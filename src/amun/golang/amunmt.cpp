#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <boost/timer/timer.hpp>
#include <boost/thread/tss.hpp>

#include "common/god.h"
#include "common/logging.h"
#include "common/threadpool.h"
#include "common/search.h"
#include "common/printer.h"
#include "common/sentence.h"
#include "common/sentences.h"
#include "common/exception.h"
#include "common/translation_task.h"

#include "amunmt.h"

using namespace amunmt;
using namespace std;

God god_;

extern "C" void init(char* options) {
  std::string optionss(options);

  god_.Init(optionss);
}


extern "C" char** translate(char** in)
{
  size_t miniSize = god_.Get<size_t>("mini-batch");
  size_t maxiSize = god_.Get<size_t>("maxi-batch");
  int miniWords = god_.Get<int>("mini-batch-words");

  std::vector<std::future< std::shared_ptr<Histories> >> results;
  SentencesPtr maxiBatch(new Sentences());

  for(int lineNum = 0; in[lineNum] != NULL; ++lineNum) {
    std::string line = in[lineNum];
    //cerr << "line=" << line << endl;

    maxiBatch->push_back(SentencePtr(new Sentence(god_, lineNum, line)));

    if (maxiBatch->size() >= maxiSize) {

      maxiBatch->SortByLength();
      while (maxiBatch->size()) {
        SentencesPtr miniBatch = maxiBatch->NextMiniBatch(miniSize, miniWords);

        results.emplace_back(
          god_.GetThreadPool().enqueue(
              [miniBatch]{ return TranslationTask(::god_, miniBatch); }
              )
        );
      }

      maxiBatch.reset(new Sentences());
    }
  }

  // last batch
  if (maxiBatch->size()) {
    maxiBatch->SortByLength();
    while (maxiBatch->size()) {
      SentencesPtr miniBatch = maxiBatch->NextMiniBatch(miniSize, miniWords);
      results.emplace_back(
        god_.GetThreadPool().enqueue(
            [miniBatch]{ return TranslationTask(::god_, miniBatch); }
            )
      );
    }
  }

  // resort batch into line number order
  Histories allHistories;
  for (auto&& result : results) {
    std::shared_ptr<Histories> histories = result.get();
    allHistories.Append(*histories);
  }
  allHistories.SortByLineNum();

  // output
  char** output = (char**)malloc(sizeof(char*) * (allHistories.size() + 1));
  for (size_t i = 0; i < allHistories.size(); ++i) {
    const History& history = *allHistories.at(i).get();
    std::stringstream ss;
    Printer(god_, history, ss);
    string str = ss.str();

    output[i] = &str[0];
  }

  return output;
}
