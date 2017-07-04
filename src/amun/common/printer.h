#pragma once

#include <vector>
#include <iomanip>

#include "common/god.h"
#include "common/history.h"
#include "common/utils.h"
#include "common/vocab.h"
#include "common/soft_alignment.h"

namespace amunmt {

std::vector<size_t> GetAlignment(const HypothesisPtr& hypothesis);

std::string GetAlignmentString(const std::vector<size_t>& alignment);
std::string GetSoftAlignmentString(const HypothesisPtr& hypothesis);

void Printer(const God &god, const History& history, std::ostream& out);
void Printer(const God &god, const Histories& histories, std::ostream& out);

}

