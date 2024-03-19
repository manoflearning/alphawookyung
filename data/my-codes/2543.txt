#include <bits/stdc++.h>
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
using namespace std;

const int INF = 1e9 + 7;

struct LR {
	int L, R;
};

struct cmp {
	bool operator() (LR a, LR b) {
		if (a.R - a.L != b.R - b.L)
			return a.R - a.L < b.R - b.L;
		else return a.L > b.L;
	}
};

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		int N; cin >> N;

		vector<int> arr(N + 1);

		priority_queue<LR, vector<LR>, cmp> pq;
		pq.push({ 1, N });

		int cnt = 0;
		while (!pq.empty()) {
			int L = pq.top().L, R = pq.top().R;
			pq.pop();

			int mid = (L + R) / 2;
			arr[mid] = ++cnt;

			if (L < mid) pq.push({ L, mid - 1 });
			if (mid < R) pq.push({ mid + 1, R });
		}

		for (int i = 1; i <= N; i++) 
			cout << arr[i] << ' ';
		cout << '\n';
	}

	return 0;
}