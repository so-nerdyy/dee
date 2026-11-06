// dee/json_min.cpp — minimal JSON parser.
#include "dee/json_min.h"
#include <cctype>
#include <stdexcept>

namespace dee {
namespace json {

namespace {

struct Parser {
    const std::string& s;
    size_t i = 0;
    bool error = false;

    Parser(const std::string& str) : s(str) {}

    void skip_ws() { while (i < s.size() && std::isspace((unsigned char)s[i])) ++i; }

    char peek() { return i < s.size() ? s[i] : '\0'; }

    ValuePtr parse_value() {
        skip_ws();
        char c = peek();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') { auto v = mk(Value::Str); v->s = parse_string(); return v; }
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') return parse_null();
        if (c == '-' || std::isdigit((unsigned char)c)) return parse_number();
        error = true; return mk(Value::Null);
    }

    ValuePtr mk(Value::Type t) { auto v = std::make_shared<Value>(); v->type = t; return v; }

    ValuePtr parse_object() {
        auto v = mk(Value::Object);
        ++i; // {
        skip_ws();
        if (peek() == '}') { ++i; return v; }
        while (true) {
            skip_ws();
            if (peek() != '"') { error = true; return v; }
            std::string key = parse_string();
            skip_ws();
            if (peek() != ':') { error = true; return v; }
            ++i; // :
            ValuePtr val = parse_value();
            v->obj[key] = val;
            skip_ws();
            char c = peek();
            if (c == ',') { ++i; continue; }
            if (c == '}') { ++i; break; }
            error = true; break;
        }
        return v;
    }

    ValuePtr parse_array() {
        auto v = mk(Value::Array);
        ++i; // [
        skip_ws();
        if (peek() == ']') { ++i; return v; }
        while (true) {
            ValuePtr val = parse_value();
            v->arr.push_back(val);
            skip_ws();
            char c = peek();
            if (c == ',') { ++i; continue; }
            if (c == ']') { ++i; break; }
            error = true; break;
        }
        return v;
    }

    std::string parse_string() {
        ++i; // opening quote
        std::string out;
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\') {
                ++i;
                char e = s[i++];
                switch (e) {
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    default: out += e; break;
                }
            } else {
                out += s[i++];
            }
        }
        if (i < s.size()) ++i; // closing quote
        return out;
    }

    ValuePtr parse_bool() {
        if (s.compare(i, 4, "true") == 0) { i += 4; auto v = mk(Value::Bool); v->b = true; return v; }
        if (s.compare(i, 5, "false") == 0) { i += 5; auto v = mk(Value::Bool); v->b = false; return v; }
        error = true; return mk(Value::Null);
    }

    // A `null` literal produces a Null-typed value (see json_min.h for the
    // convention).  The literal must end on a token boundary so malformed
    // inputs like `nul` (short) or `nulll`/`nullx` (suffixed) are rejected
    // even at the top level, where parse() does not check for trailing
    // garbage; inside arrays/objects the container's ,/]/} check would
    // already reject such a suffix.
    ValuePtr parse_null() {
        if (s.compare(i, 4, "null") != 0) {
            error = true; return mk(Value::Null);
        }
        if (i + 4 < s.size() &&
            (std::isalnum((unsigned char)s[i + 4]) || s[i + 4] == '_')) {
            error = true; return mk(Value::Null);
        }
        i += 4;
        return mk(Value::Null);
    }

    ValuePtr parse_number() {
        size_t start = i;
        if (peek() == '-') ++i;
        while (i < s.size() && (std::isdigit((unsigned char)s[i]) || s[i] == '.' ||
                                 s[i] == 'e' || s[i] == 'E' || s[i] == '+' || s[i] == '-')) ++i;
        std::string num = s.substr(start, i - start);
        bool is_float = num.find('.') != std::string::npos || num.find('e') != std::string::npos ||
                        num.find('E') != std::string::npos;
        auto v = mk(is_float ? Value::Float : Value::Int);
        if (is_float) v->f = std::stod(num); else v->i = std::stoll(num);
        return v;
    }
};

} // namespace

ValuePtr parse(const std::string& text, bool* ok) {
    Parser p(text);
    ValuePtr v = p.parse_value();
    if (ok) *ok = !p.error;
    return v;
}

} // namespace json
} // namespace dee
