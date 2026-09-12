// tests/test_json_min.cpp
//
// Unit test for the minimal JSON reader (dee/json_min.h), focused on the
// `null` literal support added for dee4-v4-segmented metadata.json files
// (p3_builder.build_segmented emits "data_file": null).
//
// Build (standalone, no cmake needed):
//   g++ -std=c++17 -I../include test_json_min.cpp ../src/json_min.cpp -o test_json_min
//
// Run:
//   ./test_json_min

#include "dee/json_min.h"

#include <cstdio>
#include <string>

using dee::json::Value;
using dee::json::ValuePtr;

static int g_checks = 0;
static int g_failures = 0;

static void check(const char* what, bool cond) {
    ++g_checks;
    printf("  [%s] %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) ++g_failures;
}

static ValuePtr parse_ok(const std::string& text, const char* what) {
    bool ok = false;
    ValuePtr v = dee::json::parse(text, &ok);
    check(what, ok && v);
    return v;
}

static void parse_err(const std::string& text, const char* what) {
    bool ok = true;
    dee::json::parse(text, &ok);
    check(what, !ok);
}

int main() {
    printf("=== json_min `null` literal test ===\n");

    // -- null as an object value (the dee4-v4-segmented metadata shape) --
    {
        ValuePtr root = parse_ok(
            R"({"format":"dee4-v4-segmented","data_file":null,"num_layers":3})",
            "object with null member parses");
        if (root && root->is_object()) {
            const Value* df = root->find("data_file");
            check("find() returns non-null for present-but-null key", df != nullptr);
            check("null member is_null()", df && df->is_null());
            check("null member is not a string", df && !df->is_string());
            check("null member is not an int", df && !df->is_int());
            check("null member is not an array", df && !df->is_array());
            check("null member is not an object", df && !df->is_object());
            check("missing key still returns nullptr", root->find("missing") == nullptr);
            const Value* n = root->find("num_layers");
            check("sibling int still parses", n && n->is_int() && n->i == 3);
        } else {
            check("root is object", false);
        }
    }

    // -- null inside arrays ---------------------------------------------
    {
        ValuePtr root = parse_ok(R"([1,null,"x",null])",
                                 "array containing nulls parses");
        if (root && root->is_array() && root->arr.size() == 4) {
            check("arr[0] is int", root->arr[0]->is_int() && root->arr[0]->i == 1);
            check("arr[1] is null", root->arr[1]->is_null());
            check("arr[2] is string", root->arr[2]->is_string() && root->arr[2]->s == "x");
            check("arr[3] is null", root->arr[3]->is_null());
        } else {
            check("root is a 4-element array", false);
        }
    }

    // -- nested null -----------------------------------------------------
    {
        ValuePtr root = parse_ok(R"({"a":{"b":[null,{"c":null}]}})",
                                 "deeply nested null parses");
        const Value* a = (root && root->is_object()) ? root->find("a") : nullptr;
        const Value* b = (a && a->is_object()) ? a->find("b") : nullptr;
        const bool ok = b && b->is_array() && b->arr.size() == 2;
        check("nested array[0] is null", ok && b->arr[0]->is_null());
        check("nested object member is null",
              ok && b->arr[1]->is_object() &&
              b->arr[1]->find("c") && b->arr[1]->find("c")->is_null());
    }

    // -- top-level null --------------------------------------------------
    {
        ValuePtr v = parse_ok("null", "top-level null parses");
        check("top-level null is_null()", v && v->is_null());
        ValuePtr vw = parse_ok("  null  ", "top-level null with whitespace parses");
        check("whitespace-padded null is_null()", vw && vw->is_null());
    }

    // -- malformed literals must be rejected ------------------------------
    parse_err("nul", "short literal 'nul' rejected");
    parse_err("nu", "short literal 'nu' rejected");
    parse_err("n", "bare 'n' rejected");
    parse_err("nulll", "suffixed literal 'nulll' rejected");
    parse_err("nullx", "suffixed literal 'nullx' rejected");
    parse_err("nullify", "suffixed literal 'nullify' rejected");
    parse_err("NULL", "uppercase 'NULL' rejected");
    parse_err("Null", "mixed-case 'Null' rejected");
    parse_err(R"({"a": nul})", "object member 'nul' rejected");
    parse_err(R"({"a": nulll})", "object member 'nulll' rejected");
    parse_err("[nul]", "array element 'nul' rejected");
    parse_err("[nullx]", "array element 'nullx' rejected");
    parse_err("[null null]", "missing comma between nulls rejected");
    parse_err(R"({"a": null "b": 1})", "missing comma after null rejected");
    parse_err("", "empty input rejected");

    // -- strings that merely look like null ------------------------------
    {
        ValuePtr root = parse_ok(R"({"a":"null","null":1})",
                                 "string \"null\" and key \"null\" parse");
        if (root && root->is_object()) {
            const Value* a = root->find("a");
            check("string \"null\" stays Str", a && a->is_string() && a->s == "null");
            const Value* k = root->find("null");
            check("key \"null\" holds an int", k && k->is_int() && k->i == 1);
        } else {
            check("root is object", false);
        }
    }

    // -- existing scalar behavior unaffected ------------------------------
    {
        ValuePtr root = parse_ok(
            R"({"t":true,"f":false,"i":-42,"x":1.5,"s":"hi","e":""})",
            "true/false/int/float/string object parses");
        if (root && root->is_object()) {
            const Value* t = root->find("t");
            const Value* f = root->find("f");
            const Value* i = root->find("i");
            const Value* x = root->find("x");
            const Value* s = root->find("s");
            const Value* e = root->find("e");
            check("true member", t && t->type == Value::Bool && t->b);
            check("false member", f && f->type == Value::Bool && !f->b);
            check("int member", i && i->is_int() && i->i == -42);
            check("float member", x && x->type == Value::Float && x->f == 1.5);
            check("string member", s && s->is_string() && s->s == "hi");
            check("empty string member", e && e->is_string() && e->s.empty());
        } else {
            check("root is object", false);
        }
        parse_err("tru", "short literal 'tru' still rejected");
        parse_err("[1,]", "trailing comma still rejected");
        parse_err("{]", "mismatched brackets still rejected");
    }

    // -- null inside a realistic segmented-store metadata excerpt ---------
    {
        const std::string meta = R"JSON({
  "format": "dee4-v4-segmented",
  "codec": "deepseek-fp4-e2m1-e8m0",
  "data_file": null,
  "integrity_file": "segments.sha256",
  "start_layer": 0,
  "num_layers": 46,
  "experts_per_layer": 256,
  "record_bytes": 8192,
  "weight_offsets": [0, 4096, 6144],
  "segments": [
    {"file": "segments/seg-000.dee4", "bucket": 0},
    {"file": "segments/seg-001.dee4", "bucket": 1}
  ]
})JSON";
        ValuePtr root = parse_ok(meta, "segmented metadata excerpt parses");
        if (root && root->is_object()) {
            check("data_file is null", root->find("data_file") &&
                  root->find("data_file")->is_null());
            const Value* segs = root->find("segments");
            const Value* bucket =
                (segs && segs->is_array() && !segs->arr.empty() &&
                 segs->arr[0] && segs->arr[0]->is_object())
                    ? segs->arr[0]->find("bucket") : nullptr;
            check("segments array intact",
                  segs && segs->is_array() && segs->arr.size() == 2 &&
                  bucket && bucket->is_int());
        } else {
            check("metadata root is object", false);
        }
    }

    printf("=== %d checks, %d failures -> %s ===\n", g_checks, g_failures,
           g_failures == 0 ? "ALL PASS" : "FAILURES");
    return g_failures == 0 ? 0 : 1;
}
