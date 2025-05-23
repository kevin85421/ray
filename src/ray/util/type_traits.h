// Copyright 2024 The Ray Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <type_traits>

namespace ray {

template <typename T>
constexpr bool AlwaysFalse = false;

template <typename T>
constexpr bool AlwaysTrue = true;

template <int N>
constexpr bool AlwaysFalseValue = false;

template <int N>
constexpr bool AlwaysTrueValue = true;

template <typename, typename = void>
struct has_equal_operator : std::false_type {};

template <typename T>
struct has_equal_operator<T,
                          std::void_t<decltype(std::declval<T>() == std::declval<T>())>>
    : std::true_type {};

}  // namespace ray
