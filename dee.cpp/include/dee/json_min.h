// dee/json_min.h
//
// A tiny, dependency-free JSON reader sufficient for safetensors headers
// (objects, arrays, strings, integers). Not a general-purpose JSON library.
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

    const Value* find(const std::string& key) const {
        if (type != Object) return nullptr;
        auto it = obj.find(key);
        return it == obj.end() ? nullptr : it->second.get();
    }
};

// Parse `text`. On error returns a Value of type Null and sets *ok = false.
ValuePtr parse(const std::string& text, bool* ok = nullptr);

} // namespace json
} // namespace dee
