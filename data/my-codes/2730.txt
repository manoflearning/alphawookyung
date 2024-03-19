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
const int dy[] = { 0, 0, 0, 1, -1, 1, 1, -1, -1 };
const int dx[] = { 0, 1, -1, 0, 0, 1, -1, 1, -1 };

class Solution {
public:
    vector<vector<int>> resultGrid(vector<vector<int>>& a, int threshold) {
        int n = sz(a), m = sz(a[0]);
        
        vector<vector<int>> avg(n, vector<int>(m, -1));
        for (int i = 1; i < n - 1; i++) {
            for (int j = 1; j < m - 1; j++) {
                int diff = 0;
                
                diff = max(diff, abs(a[i - 1][j - 1] - a[i - 1][j]));
                diff = max(diff, abs(a[i - 1][j + 1] - a[i - 1][j]));
                diff = max(diff, abs(a[i][j - 1] - a[i][j]));
                diff = max(diff, abs(a[i][j + 1] - a[i][j]));
                diff = max(diff, abs(a[i + 1][j - 1] - a[i + 1][j]));
                diff = max(diff, abs(a[i + 1][j + 1] - a[i + 1][j]));
                
                diff = max(diff, abs(a[i - 1][j - 1] - a[i][j - 1]));
                diff = max(diff, abs(a[i + 1][j - 1] - a[i][j - 1]));
                diff = max(diff, abs(a[i - 1][j] - a[i][j]));
                diff = max(diff, abs(a[i + 1][j] - a[i][j]));
                diff = max(diff, abs(a[i - 1][j + 1] - a[i][j + 1]));
                diff = max(diff, abs(a[i + 1][j + 1] - a[i][j + 1]));
                
                if (diff <= threshold) {
                    int sum = 0;
                    for (int d = 0; d < 9; d++)
                        sum += a[i + dy[d]][j + dx[d]];
                    avg[i][j] = sum / 9;
                }
            }
        }
        
        vector<vector<int>> ret(n, vector<int>(m, -1));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                int sum = 0, cnt = 0;
                for (int d = 0; d < 9; d++) {
                    int y = i + dy[d], x = j + dx[d];
                    if (y < 0 || n <= y || x < 0 || m <= x) continue;
                    if (avg[y][x] == -1) continue;
                    sum += avg[y][x];
                    cnt++;
                }
                if (cnt == 0) ret[i][j] = a[i][j];
                else ret[i][j] = sum / cnt;
            }
        }
        
        return ret;
    }
};