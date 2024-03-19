#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
#include <cassert>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

struct dw {
    int d, w;
};

struct cmp {
    bool operator()(const dw& a, const dw& b) {
        return a.d < b.d;
    }
};

struct cmp2 {
    bool operator()(const dw& a, const dw& b) {
        return a.w < b.w;
    } 
};

priority_queue<dw, vector<dw>, cmp> pq;
priority_queue<dw, vector<dw>, cmp2> pq2;

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);

    int N; cin>>N;
    for(int i=0; i<N; i++) {
        int d, w;
        cin>>d>>w;
        pq.push({d, w});
    }

    int res = 0;
    for (int i=N; i>=1; i--) {
        while(!pq.empty() && pq.top().d >= i) {
            pq2.push(pq.top());
            pq.pop();
        }
        if(pq2.empty()) continue;
        res += pq2.top().w;
        pq2.pop();
    }

    cout<<res;

    return 0;
}