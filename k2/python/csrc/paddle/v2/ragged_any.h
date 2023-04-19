/**
 * @brief A wrapper around Ragged<Any> and paddle::Tensor
 *
 * @copyright
 * Copyright      2022  PaddlePaddle.    (author: Hui Zhang)
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

#ifndef K2_PYTHON_CSRC_PADDLE_V2_RAGGED_ANY_H_
#define K2_PYTHON_CSRC_PADDLE_V2_RAGGED_ANY_H_
#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "k2/csrc/ragged.h"
#include "k2/python/csrc/paddle.h"

// #include "paddle/utils/pybind.h"

namespace pybind11 {
namespace detail {
  
// Pybind11 bindings for optional types.
// http://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html#c-17-library-containers
template <typename T>
struct type_caster<paddle::optional<T>> : optional_caster<paddle::optional<T>> {
};

}  // namespace detail
}  // namespace pybind11

namespace k2 {

/** RaggedAny is introduced to support backward propagation on
 Ragged<Any> since there has to be a tensor involved during backward
 propagations.

 Ragged<Any> unifies Ragged<int32_t>, Ragged<float>, and Ragged<double>
 so that Python only sees Ragged<Any>.
*/
struct RaggedAny {
  Ragged<Any> any;
  paddle::Tensor data;  //!< shares the underlying memory with any.values

  /// The default constructor initializes an invalid ragged tensor.
  RaggedAny() = default;
  /** Construct a ragged tensor from a shape and a value.

     @param shape The shape of the ragged tensor.
     @param value 1-D tensor containing the value of the ragged tensor.
   */
  RaggedAny(const RaggedShape &shape, paddle::Tensor value);

  /* Create a ragged tensor from a paddle tensor.

     @note The resulting ragged tensor has a regular structure.

     @params tensor An N-D Paddle tensor, where N > 1. Supported dtypes are
                    paddle.int32, paddle.float32, paddle.float64.

     @caution If the input tensor is contiguous, the ragged tensor shares the
     underlying memory with the input tensor. Otherwise, memory is copied.
   */
  explicit RaggedAny(paddle::Tensor tensor);

  RaggedAny(const RaggedAny &) = default;
  RaggedAny &operator=(const RaggedAny &) = default;
  RaggedAny(RaggedAny &&) = default;
  RaggedAny &operator=(RaggedAny &&) = default;

  explicit RaggedAny(const Ragged<Any> &any) : any(any) {}

  /** Create a ragged tensor from its string representation.

      An example string with 3 axes is::

      [ [[1 2] [3] []]   [[1] [10] [20 30]] ]

     @param s  The string representation of a ragged tensor.
     @param dtype  An instance of `paddle.dtype`. Supported dtypes are:
                   `paddle.float32`, `paddle.float64`, and
                   `paddle.int32`. If it is `None`, the dtype is
                   inferred from the given string.  It first tries
                   to use `paddle.int32`. If it fails, it will switch
                   to `paddle.float32`.

     @note We can support other dtypes if needed.
   */
  explicit RaggedAny(const std::string &s, py::object dtype = py::none(),
                     paddle::Place device = paddle::Place(paddle::AllocationType::CPU));

  explicit RaggedAny(const std::string &s, py::object dtype = py::none(),
                     const std::string &device = "cpu")
      : RaggedAny(s, dtype, paddle::Place(device)) {}

  /** Create a ragged tensor from a list of sublist(s).

     @param data A python list-of lists.
     @param dtype An instance of `paddle.dtype`. If it is `None`,
                  the data type is inferred from the input `data`,
                  which will either be paddle.int32 or paddle.float32.
                  Supported dtypes are: `paddle.int32`, paddle.`float32`,
                  and `paddle.float64`.

     @note It supports `data` with number of axes >= 2.
   */
  explicit RaggedAny(py::list data, py::object dtype = py::none(),
                     paddle::Place device = paddle::Place(paddle::AllocationType::CPU));

  explicit RaggedAny(py::list data, py::object dtype = py::none(),
                     const std::string device = "cpu")
      : RaggedAny(data, dtype, paddle::Place(device)) {}

  /// Populate `this->data` and return it
  const paddle::Tensor &Data() const;

  /** Convert a ragged tensor to a string.

     An example output for ``compact==false``:

        RaggedTensor([[[1, 2, 3],
                       [],
                       [0]],
                      [[2],
                       [3, 10.5]]], dtype=paddle.float32)

     An example output for ``compact==true``:

     RaggedTensor([[[1, 2, 3], [], [0]], [[2], [3, 10.5]]], dtype=paddle.float32)

     @param device_id -1 for CPU. 0 and above is for CUDA.
     @param compact If false, each sublist occupies a row. If true, all sublists
                    occupies only one row.
     @return Return a string representation of this tensor.
   */
  std::string ToString(bool compact = false, int device_id = -1) const;

  /* Move a ragged tensor to a given device.

     Note: If the this tensor is already on the given device, itself
     is returned. Otherwise, a copy of the this tensor moved to the given
     device is returned.

     @param device  A paddle device, which can be either a CPU device
                    or a CUDA device.

     @return Return a ragged tensor on the given device.
   */
  RaggedAny To(paddle::Place device) const;

  /** Move this tensor to a given device.
*
    Note: If the this tensor is already on the given device, itself
    is returned. Otherwise, a copy of the this tensor moved to the given
    device is returned.

    @param device A string representation of a device, e.g.,
                  "cpu", "cuda:0", "cuda:1", etc.

    @return Return a ragged tensor on the given device.
   */
  RaggedAny To(const std::string &device) const;

  /* Convert a ragged tensor to given scalar type.

     Note: If the this tensor is already of the given type, itself
     is returned. Otherwise, a copy of the this tensor converted to the given
     type is returned.

     @param scalar_type The type this tensor should be converted to.

     @return Return a ragged tensor with the specified type.
   */
  RaggedAny To(paddle::DataType scalar_type) const;

  /// Return a copy of this ragged tensor
  RaggedAny Clone() const;

  /** Enable/Disable requires_grad of this tensor

     @param requires_grad True to require grad for this tensors.
                          False to not require grad.

     @note If this is NOT a float tenor and requires_grad is `True`,
     Paddle will throw a RuntimeError exception.
   */
  RaggedAny &SetRequiresGrad(bool requires_grad = true);

  /** Compute the sum over the last axis of the ragged tensor.

     It is a wrapper around k2::SumPerSublist.

     @note It supports autograd if the dtype of this tensor is
     `paddle.float32` or `paddle.float64`.

     @param initial_value  This value is added to the sum of each
     sub-list. If a sublist is empty, the sum of it is just initial_value.

     @return Return the sum of each sublist as a 1-D tensor.
   */
  paddle::Tensor Sum(float initial_value = 0) const;

  /** Compute the logsumexp over the last axis of the ragged tensor.

     It is a wrapper around k2::LogSumPerSublist.

     @note It only accepts input with dtype
     `paddle.float32` or `paddle.float64`.

     @param initial_value If a sublist is empty,
     the logsumexp of it is just initial_value.

     @return Return the logsumexp of each sublist as a 1-D tensor.
   */
  paddle::Tensor LogSumExp(
      float initial_value = -std::numeric_limits<float>::infinity()) const;

  /** Index a ragged tensor (supporting only axis==0 at present).

     It requires that the ragged tensor has at least 3 axes.

     @TODO: To add autograd support.

     @param axis  The axis to index. Must be 0 at present.
     @param i  The i-th sublist of the specified axis.

     @return Return a ragged tensor with one fewer axis.
     It shares data with "this" tensor.
   */
  RaggedAny Index(int32_t axis, int32_t i) const;

  /** A wrapper around k2::RemoveAxis. See its doc in
     k2/csrc/ragged_ops.h
   */
  RaggedAny RemoveAxis(int32_t axis) /*const*/;

  /** A wrapper for k2::RaggedArange. See its doc for help.
   */
  RaggedAny Arange(int32_t axis, int32_t begin, int32_t end) /*const*/;

  /// Wrapper for k2::RemoveValuesLeq()
  RaggedAny RemoveValuesLeq(py::object cutoff) /*const*/;

  /// Wrapper for k2::RemoveValuesEq()
  RaggedAny RemoveValuesEq(py::object target) /*const*/;

  /// Wrapper for k2::ArgMaxPerSublist
  paddle::Tensor ArgMax(py::object initial_value = py::none()) /*const*/;

  // Wrapper for k2::MaxPerSublist
  paddle::Tensor Max(py::object initial_value = py::none()) /*const*/;

  // Wrapper for k2::MinPerSublist
  paddle::Tensor Min(py::object initial_value) /*const*/;

  /// Wrapper for k2::Cat
  static RaggedAny Cat(const std::vector<RaggedAny> &srcs, int32_t axis);

  /// Wrapper for k2::UniqueSequences
  std::tuple<RaggedAny, paddle::optional<RaggedAny>,
             paddle::optional<paddle::Tensor>>
  Unique(bool need_num_repeats = false, bool need_new2old_indexes = false);

  /// Wrapper for k2::NormalizePerSublist
  RaggedAny Normalize(bool use_log) /*const*/;

  RaggedAny Add(paddle::Tensor value, py::object alpha) /*const*/;

  /// Wrapper for k2::PadRagged
  paddle::Tensor Pad(const std::string &mode,
                    py::object padding_value) /*const*/;

  /// Convert a ragged tensor to a list of lists [of lists ...]
  /// Note: You can use the return list to construct a ragged tensor.
  py::list ToList() /*const*/;

  /// Wrapper for k2::SortSublists
  paddle::optional<paddle::Tensor> Sort(bool descending = false,
                                      bool need_new2old_indexes = false);

  /// Wrapper for k2::Index
  RaggedAny Index(RaggedAny &indexes) /*const*/;

  /// Wrapper for k2::Index
  std::pair<RaggedAny, paddle::optional<paddle::Tensor>> Index(
      paddle::Tensor indexes, int32_t axis,
      bool need_value_indexes = false) /*const*/;

  /// Wrapper for k2::Index
  RaggedAny Index(paddle::Tensor src,
                  py::object default_value = py::none()) /*const*/;

  /// Wrapper for k2::Index
  paddle::Tensor IndexAndSum(paddle::Tensor src) /*const*/;
};

}  // namespace k2

#endif  // K2_PYTHON_CSRC_PADDLE_V2_RAGGED_ANY_H_
