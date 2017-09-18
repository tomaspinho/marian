#include "matrix.h"

namespace amunmt {
namespace GPU {
namespace mblas {

std::ostream& operator<<(std::ostream& os, const half &val)
{
  os << "half";
  return os;
}

}
}
}
