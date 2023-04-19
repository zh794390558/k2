/**
 * @brief Everything related to PaddlePaddle for k2 Python wrappers.
 *
 * @copyright
 * Copyright      2023  PaddlePaddle.        (authors: Hui Zhang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "k2/python/csrc/paddle.h"

#if defined(K2_USE_PADDLE)

#include "k2/python/csrc/paddle/arc.h"
#include "k2/python/csrc/paddle/fsa.h"
// #include "k2/python/csrc/paddle/fsa_algo.h"
// #include "k2/python/csrc/paddle/index_add.h"
// #include "k2/python/csrc/paddle/index_select.h"
// #include "k2/python/csrc/paddle/mutual_information.h"
// #include "k2/python/csrc/paddle/nbest.h"
// #include "k2/python/csrc/paddle/ragged.h"
// #include "k2/python/csrc/paddle/ragged_ops.h"
// #include "k2/python/csrc/paddle/rnnt_decode.h"
// #include "k2/python/csrc/paddle/v2/k2.h"

void PybindPaddle(py::module &m) {
  PybindArc(m);
  PybindFsa(m);
  // PybindFsaAlgo(m);
  // PybindIndexAdd(m);
  // PybindIndexSelect(m);
  // PybindMutualInformation(m);
  // PybindNbest(m);
  // PybindRagged(m);
  // PybindRaggedOps(m);
  // PybindRnntDecode(m);

  // k2::PybindV2(m);
}

#else

void PybindPaddle(py::module &) {}

#endif
