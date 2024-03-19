#include <iostream>
#include <vector>
#include <queue>
#include <cstring>
using namespace std;

const int INF = 1e9 + 7;
const int MAXV = 4 * 1e5;

struct friends {
    int x, Kx, y, Ky;
};

int N, A[MAXV + 5], B[MAXV + 5], dist[MAXV + 5];
bool used[MAXV + 5];
vector<int> adj[MAXV + 5];
friends arr[MAXV + 5];

void input();
int bSearch(int L, int R);
void bInit();
int hopcroft();
void bfs();
bool dfs(int a);

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);

    input();
    
    int ans = bSearch(0, 1e6 + 1);

    if (ans == 1e6 + 1) cout << -1;
    else cout << ans;

    return 0;
}

void input() {
    cin >> N;
    for (int i = 0; i < N; i++) {
        cin >> arr[i].x >> arr[i].Kx >> arr[i].y >> arr[i].Ky;
    }
}

int bSearch(int L, int R) {
    while (L < R) {
        bInit();

        int mid = (L + R) / 2;
        
        for (int i = 0; i < N; i++) {
            if (arr[i].Kx <= mid)
                adj[i].push_back(arr[i].x);
            if (arr[i].Ky <= mid)
                adj[i].push_back(arr[i].y);
        }

        int total = hopcroft();
        
        if (total == N) R = mid;
        else L = mid + 1;
    }
    return L;
}

void bInit() {
    memset(A, -1, sizeof(A));
    memset(B, -1, sizeof(B));
    memset(used, 0, sizeof(used));

    for (int i = 0; i < MAXV + 5; i++) adj[i].clear();
}

int hopcroft() {
    int ret = 0;
    while (true) {
        bfs();

        int flow = 0;
        for (int i = 0; i < N; i++)
            if (!used[i] && dfs(i)) flow++;

        if (flow == 0) break;
        ret += flow;
    }
    return ret;
}

void bfs() {
    queue<int> q;
    for (int i = 0; i < N; i++) {
        if (!used[i]) {
            dist[i] = 0;
            q.push(i);
        }
        else dist[i] = INF;
    }

    while (!q.empty()) {
        int a = q.front();
        q.pop();
        for (int b : adj[a]) {
            if (B[b] != -1 && dist[B[b]] == INF) {
                dist[B[b]] = dist[a] + 1;
                q.push(B[b]);
            }
        }
    }
}

bool dfs(int a) {
    for (int b : adj[a]) {
        if (B[b] == -1 || (dist[B[b]] == dist[a] + 1 && dfs(B[b]))) {
            used[a] = true;
            A[a] = b;
            B[b] = a;
            return true;
        }
    }
    return false;
}