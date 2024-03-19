#define ll long long
#define ull unsigned long long
#define ld long double
#define pii pair<int, int>
#define pll pair<ll, ll>
#define fr first
#define sc second
#define all(c) (c).begin(), (c).end()
#define sz(x) (int)(x).size()

const double EPS = 1e-9;
const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1, 1, 1, -1, -1 };
const int dx[] = { 1, -1, 0, 0, 1, -1, 1, -1 };

class Solution {
public:
    int minimumTimeToInitialState(string word, int k) {
        int ret = (sz(word) + k - 1) / k;
        for (int i = k; i < sz(word); i += k) {
            bool isValid = 1;
            for (int j = i, p = 0; j < sz(word); j++, p++) {
                if (word[j] != word[p]) {
                    isValid = 0;
                    break;
                }
            }
            if (isValid) { ret = i / k; break; }
        }
        return ret;
    }
};