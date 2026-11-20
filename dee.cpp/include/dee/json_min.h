// dee/json_min.h
//
// A tiny, dependency-free JSON reader sufficient for safetensors headers
// (objects, arrays, strings, integers, floats, booleans, and the `null`
// literal). Not a general-purpose JSON library.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace dee {
namespace json {

struct Value;
using ValuePtr = std::shared_ptr<Value>;

struct Value {
    enum Type { Null, Bool, Int, Float, Str, Array, Object } type = Null;
    bool        b = false;
    long long   i = 0;
    double      f = 0.0;
    std::string s;
    std::vector<ValuePtr> arr;
    std::map<std::string, ValuePtr> obj;

    bool is_object() const { return type == Object; }
    bool is_array()  const { return type == Array; }
    bool is_string() const { return type == Str; }
    bool is_int()    const { return type == Int; }
    bool is_null()   const { return type == Null; }

    const Value* find(const std::string& key) const {
        if (type != Object) return nullptr;
        auto it = obj.find(key);
        return it == obj.end() ? nullptr : it->second.get();
    }
};

// Parse `text`. On error returns a Value of type Null and sets *ok = false.
// A JSON `null` literal also produces a Value of type Null (is_null()), so
// use *ok — not the value's type — to detect parse failure.  find() returns
// non-null for a present-but-null key and nullptr for a missing key, and
// every other type predicate (is_string/is_int/is_array/is_object) returns
// false for Null, so typed accessors treat a null field exactly like a
// missing or wrong-typed one.
ValuePtr parse(const std::string& text, bool* ok = nullptr);

} // namespace json
} // namespace dee
