#define _CRT_NONSTDC_NO_WARNINGS
#define _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING
#include <bits/stdc++.h>
#include <random>
#include <unordered_set>
#include <array>
#ifdef _MSC_VER
#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <conio.h>
#include <ppl.h>
#include <filesystem>
#include <intrin.h>
//#include <boost/multiprecision/cpp_int.hpp>
int __builtin_clz(unsigned int n)
{
    unsigned long index;
    _BitScanReverse(&index, n);
    return 31 - index;
}
int __builtin_ctz(unsigned int n)
{
    unsigned long index;
    _BitScanForward(&index, n);
    return index;
}
namespace std {
    inline int __lg(int __n) { return sizeof(int) * 8 - 1 - __builtin_clz(__n); }
}
//using __uint128_t = boost::multiprecision::uint128_t;
#else
#pragma GCC target("avx2")
#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")
#endif

/** compro_io **/

/* tuple */
// out
namespace aux {
    template<typename T, unsigned N, unsigned L>
    struct tp {
        static void output(std::ostream& os, const T& v) {
            os << std::get<N>(v) << ", ";
            tp<T, N + 1, L>::output(os, v);
        }
    };
    template<typename T, unsigned N>
    struct tp<T, N, N> {
        static void output(std::ostream& os, const T& v) { os << std::get<N>(v); }
    };
}
template<typename... Ts>
std::ostream& operator<<(std::ostream& os, const std::tuple<Ts...>& t) {
    os << '[';
    aux::tp<std::tuple<Ts...>, 0, sizeof...(Ts) - 1>::output(os, t);
    return os << ']';
}

template<class Ch, class Tr, class Container>
std::basic_ostream<Ch, Tr>& operator<<(std::basic_ostream<Ch, Tr>& os, const Container& x);

/* pair */
// out
template<class S, class T>
std::ostream& operator<<(std::ostream& os, const std::pair<S, T>& p) {
    return os << "[" << p.first << ", " << p.second << "]";
}
// in
template<class S, class T>
std::istream& operator>>(std::istream& is, std::pair<S, T>& p) {
    return is >> p.first >> p.second;
}

/* container */
// out
template<class Ch, class Tr, class Container>
std::basic_ostream<Ch, Tr>& operator<<(std::basic_ostream<Ch, Tr>& os, const Container& x) {
    bool f = true;
    os << "[";
    for (auto& y : x) {
        os << (f ? "" : ", ") << y;
        f = false;
    }
    return os << "]";
}
// in
template <
    class T,
    class = decltype(std::begin(std::declval<T&>())),
    class = typename std::enable_if<!std::is_same<T, std::string>::value>::type
>
std::istream& operator>>(std::istream& is, T& a) {
    for (auto& x : a) is >> x;
    return is;
}

std::ostream& operator<<(std::ostream& os, const std::vector<bool>& v) {
    std::string s(v.size(), ' ');
    for (int i = 0; i < v.size(); i++) s[i] = v[i] + '0';
    os << s;
    return os;
}

/* struct */
template<typename T>
auto operator<<(std::ostream& out, const T& t) -> decltype(out << t.stringify()) {
    out << t.stringify();
    return out;
}

/* setup */
struct IOSetup {
    IOSetup(bool f) {
        if (f) { std::cin.tie(nullptr); std::ios::sync_with_stdio(false); }
        std::cout << std::fixed << std::setprecision(15);
    }
} iosetup(true);

/** string formatter **/
template<typename... Ts>
std::string format(const std::string& f, Ts... t) {
    size_t l = std::snprintf(nullptr, 0, f.c_str(), t...);
    std::vector<char> b(l + 1);
    std::snprintf(&b[0], l + 1, f.c_str(), t...);
    return std::string(&b[0], &b[0] + l);
}

template<typename T>
std::string stringify(const T& x) {
    std::ostringstream oss;
    oss << x;
    return oss.str();
}

/* dump */
#define DUMPOUT std::cerr
std::ostringstream DUMPBUF;
#define dump(...) do{DUMPBUF<<"  ";DUMPBUF<<#__VA_ARGS__<<" :[DUMP - "<<__LINE__<<":"<<__FUNCTION__<<"]"<<std::endl;DUMPBUF<<"    ";dump_func(__VA_ARGS__);DUMPOUT<<DUMPBUF.str();DUMPBUF.str("");DUMPBUF.clear();}while(0);
void dump_func() { DUMPBUF << std::endl; }
template <class Head, class... Tail> void dump_func(Head&& head, Tail&&... tail) { DUMPBUF << head; if (sizeof...(Tail) == 0) { DUMPBUF << " "; } else { DUMPBUF << ", "; } dump_func(std::move(tail)...); }

/* timer */
class Timer {
    double t = 0, paused = 0, tmp;
public:
    Timer() { reset(); }
    static double time() {
#ifdef _MSC_VER
        return __rdtsc() / 3.0e9;
#else
        unsigned long long a, d;
        __asm__ volatile("rdtsc"
            : "=a"(a), "=d"(d));
        return (d << 32 | a) / 3.0e9;
#endif
    }
    void reset() { t = time(); }
    void pause() { tmp = time(); }
    void restart() { paused += time() - tmp; }
    double elapsed_ms() const { return (time() - t - paused) * 1000.0; }
};

/* rand */
struct Xorshift {
    static constexpr uint64_t M = INT_MAX;
    static constexpr double e = 1.0 / M;
    uint64_t x = 88172645463325252LL;
    Xorshift() {}
    Xorshift(uint64_t seed) { reseed(seed); }
    inline void reseed(uint64_t seed) { x = 0x498b3bc5 ^ seed; for (int i = 0; i < 20; i++) next(); }
    inline uint64_t next() { x = x ^ (x << 7); return x = x ^ (x >> 9); }
    inline int next_int() { return next() & M; }
    inline int next_int(int mod) { return next() % mod; }
    inline int next_int(int l, int r) { return l + next_int(r - l + 1); }
    inline double next_double() { return next_int() * e; }
} rnd;

/* shuffle */
template<typename T>
void shuffle_vector(std::vector<T>& v, Xorshift& rnd) {
    int n = v.size();
    for (int i = n - 1; i >= 1; i--) {
        int r = rnd.next_int(i);
        std::swap(v[i], v[r]);
    }
}

/* split */
std::vector<std::string> split(std::string str, const std::string& delim) {
    for (char& c : str) if (delim.find(c) != std::string::npos) c = ' ';
    std::istringstream iss(str);
    std::vector<std::string> parsed;
    std::string buf;
    while (iss >> buf) parsed.push_back(buf);
    return parsed;
}

template<typename A, size_t N, typename T> inline void Fill(A(&array)[N], const T& val) {
    std::fill((T*)array, (T*)(array + N), val);
}

template<typename T, typename ...Args> auto make_vector(T x, int arg, Args ...args) { if constexpr (sizeof...(args) == 0)return std::vector<T>(arg, x); else return std::vector(arg, make_vector<T>(x, args...)); }
template<typename T> bool chmax(T& a, const T& b) { if (a < b) { a = b; return true; } return false; }
template<typename T> bool chmin(T& a, const T& b) { if (a > b) { a = b; return true; } return false; }

using ll = long long;
using ld = double;
//using ld = boost::multiprecision::cpp_bin_float_quad;
using pii = std::pair<int, int>;
using pll = std::pair<ll, ll>;

using std::cin, std::cout, std::cerr, std::endl, std::string, std::vector, std::array;



constexpr int NN = 50;
constexpr int KK = 500;
template<typename T> using Grid = array<array<T, NN>, NN>;

static constexpr char USED = -1;
static constexpr int di[4] = { 0, 1, 0, -1 };
static constexpr int dj[4] = { 1, 0, -1, 0 };

struct Input;
using InputPtr = std::shared_ptr<Input>;
struct Input {
    int N, K;
    Grid<char> grid;
    Input(std::istream& in) {
        in >> N >> K;
        memset(grid.data(), -1, sizeof(char) * NN * NN);
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                in >> grid[i][j];
                grid[i][j] -= '0';
            }
        }
    }
    string stringify() const {
        string res = format("%d %d\n", N, K);
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                res += char(grid[i][j] + '0');
            }
            res += '\n';
        }
        return res;
    }
};

struct MoveAction {
    int before_row, before_col, after_row, after_col;
    MoveAction(int before_row, int before_col, int after_row, int after_col) :
        before_row(before_row), before_col(before_col), after_row(after_row), after_col(after_col) {}
};

struct ConnectAction {
    int c1_row, c1_col, c2_row, c2_col;
    ConnectAction(int c1_row, int c1_col, int c2_row, int c2_col) :
        c1_row(c1_row), c1_col(c1_col), c2_row(c2_row), c2_col(c2_col) {}
};

struct Result {
    vector<MoveAction> move;
    vector<ConnectAction> connect;
    Result(const vector<MoveAction>& move, const vector<ConnectAction>& con) : move(move), connect(con) {}
};

struct UnionFind {
    vector<int> data;

    UnionFind() = default;

    explicit UnionFind(size_t sz) : data(sz, -1) {}

    bool unite(int x, int y) {
        x = find(x), y = find(y);
        if (x == y) return false;
        if (data[x] > data[y]) std::swap(x, y);
        data[x] += data[y];
        data[y] = x;
        return true;
    }

    int find(int k) {
        if (data[k] < 0) return (k);
        return data[k] = find(data[k]);
    }

    int size(int k) {
        return -data[find(k)];
    }

    bool same(int x, int y) {
        return find(x) == find(y);
    }

    vector<vector<int>> groups() {
        int n = (int)data.size();
        vector< vector< int > > ret(n);
        for (int i = 0; i < n; i++) {
            ret[find(i)].emplace_back(i);
        }
        ret.erase(remove_if(begin(ret), end(ret), [&](const vector< int >& v) {
            return v.empty();
            }), ret.end());
        return ret;
    }
};

struct Cell {
    int id;
    int i, j;
    int color;
    Cell* next[4];
    Cell(int id = -1, int i = -1, int j = -1, int color = -1) : id(id), i(i), j(j), color(color), next() {}
    string stringify() const {
        return format("Cell [id=%d, i=%d, j=%d, color=%d]", id, i, j, color);
    }
};

struct LinkedList2D {

    int N;
    int V;
    Cell head;
    Cell cells[KK];
    Grid<Cell*> grid;

    LinkedList2D(int N, const Grid<char>& g) : N(N), V(0), head(), cells(), grid() {
        for (auto& v : grid) std::fill(v.begin(), v.end(), &head);
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                if (g[i][j]) {
                    cells[V] = Cell(V, i, j, g[i][j]);
                    grid[i][j] = &cells[V];
                    V++;
                }
                else {
                    grid[i][j] = nullptr;
                }
            }
        }
        for (auto& cell : cells) {
            for (int d = 0; d < 4; d++) {
                cell.next[d] = search(cell.i, cell.j, d);
            }
        }
        eval();
    }

    inline Cell* search(int i, int j, int d) const {
        i += di[d]; j += dj[d];
        while (!grid[i][j]) i += di[d], j += dj[d];
        return grid[i][j];
    }

    inline Cell* search(Cell* c, int d) const {
        return search(c->i, c->j, d);
    }

    int eval() const {
        bool used[KK] = {};
        int score = 0;
        for (const auto& sc : cells) {
            if (used[sc.id]) continue;
            int nc = 0;
            std::queue<const Cell*> qu({ &sc });
            used[sc.id] = true;
            nc++;
            while (!qu.empty()) {
                auto u = qu.front(); qu.pop();
                for (int d = 0; d < 4; d++) {
                    auto v = u->next[d];
                    if (!used[v->id] && v->color == u->color) {
                        qu.push(v);
                        used[v->id] = true;
                        nc++;
                    }
                }
            }
            score += nc * nc;
            dump(nc, score);
        }
        return score;
    }

    void print() const {
        for (int i = 0; i <= N + 1; i++) {
            for (int j = 0; j <= N + 1; j++) {
                if (grid[i][j] == &head) fprintf(stderr, "#");
                else if (grid[i][j]) fprintf(stderr, "%d", grid[i][j]->color);
                else fprintf(stderr, " ");
            }
            fprintf(stderr, "\n");
        }
    }

};

struct Solver;
using SolverPtr = std::shared_ptr<Solver>;
struct Solver {

    int N, K;
    int action_count_limit;
    Grid<char> grid;

    Solver(InputPtr input) : N(input->N), K(input->K), action_count_limit(K * 100), grid(input->grid) {}

    Result solve() {

        LinkedList2D ll2d(N, grid);
        return Result{ {},{} };
    }

};

struct UnionFind_ {
    std::map<pii, pii> parent;
    UnionFind_() :parent() {}

    pii find(pii x)
    {
        if (parent.find(x) == parent.end()) {
            parent[x] = x;
            return x;
        }
        else if (parent[x] == x) {
            return x;
        }
        else {
            parent[x] = find(parent[x]);
            return parent[x];
        }
    }

    void unite(pii x, pii y)
    {
        x = find(x);
        y = find(y);
        if (x != y) {
            parent[x] = y;
        }
    }
};

int calc_score(InputPtr input, const Result& res) {

    auto N = input->N;
    auto field = input->grid;
    for (auto r : res.move) {
        assert(field[r.before_row][r.before_col]);
        assert(!field[r.after_row][r.after_col]);
        std::swap(field[r.before_row][r.before_col], field[r.after_row][r.after_col]);
    }

    UnionFind_ uf;
    for (auto r : res.connect) {
        pii p1(r.c1_row, r.c1_col), p2(r.c2_row, r.c2_col);
        uf.unite(p1, p2);
    }

    vector<pii> computers;
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            if (field[i][j]) {
                computers.emplace_back(i, j);
            }
        }
    }

    int score = 0;
    for (int i = 0; i < (int)computers.size(); i++) {
        for (int j = i + 1; j < (int)computers.size(); j++) {
            auto c1 = computers[i];
            auto c2 = computers[j];
            if (uf.find(c1) == uf.find(c2)) {
                score += (field[c1.first][c1.second] == field[c2.first][c2.second]) ? 1 : -1;
            }
        }
    }

    return std::max(score, 0);
}

void print_answer(std::ostream& out, const Result& res) {
    out << res.move.size() << endl;
    for (auto m : res.move) {
        out << m.before_row - 1 << " " << m.before_col - 1 << " "
            << m.after_row - 1 << " " << m.after_col - 1 << endl;
    }
    out << res.connect.size() << endl;
    for (auto m : res.connect) {
        out << m.c1_row - 1 << " " << m.c1_col - 1 << " "
            << m.c2_row - 1 << " " << m.c2_col - 1 << endl;
    }
}



#ifdef _MSC_VER
void batch_test(int seed_begin = 0, int num_seed = 100) {

    constexpr int batch_size = 8;
    int seed_end = seed_begin + num_seed;

    vector<int> scores(num_seed);
    concurrency::critical_section mtx;
    for (int batch_begin = seed_begin; batch_begin < seed_end; batch_begin += batch_size) {
        int batch_end = std::min(batch_begin + batch_size, seed_end);
        concurrency::parallel_for(batch_begin, batch_end, [&mtx, &scores](int seed) {
            std::ifstream ifs(format("tools/in/%04d.txt", seed));
            std::istream& in = ifs;
            std::ofstream ofs(format("tools/out/%04d.txt", seed));
            std::ostream& out = ofs;

            auto input = std::make_shared<Input>(in);
            Solver solver(input);
            auto ret = solver.solve();
            print_answer(out, ret);

            {
                mtx.lock();
                scores[seed] = calc_score(input, ret);
                cerr << seed << ": " << scores[seed] << '\n';
                mtx.unlock();
            }
        });
    }

    dump(std::accumulate(scores.begin(), scores.end(), 0));
}
#endif


int main(int argc, char** argv) {

#ifdef HAVE_OPENCV_HIGHGUI
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
#endif

#ifdef _MSC_VER
    std::ifstream ifs(R"(tools\in\0003.txt)");
    std::ofstream ofs(R"(tools\out\0003.txt)");
    std::istream& in = ifs;
    std::ostream& out = ofs;
#else
    std::istream& in = cin;
    std::ostream& out = cout;
#endif

#if 0
    batch_test();
#else
    auto input = std::make_shared<Input>(in);
    cerr << *input << endl;
    Solver solver(input);
    auto ret = solver.solve();
    dump(calc_score(input, ret));
    print_answer(out, ret);
#endif

    return 0;
}