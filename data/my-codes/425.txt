#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
using namespace std;
#define ll long long

struct S {
	ll sum, bit;
};

int N;
ll w1[41], w2[41];
vector<S> L[21], R[21];

void f(int p, int up, int cnt, ll sum, ll bit);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> N;
	for (int i = 1; i <= N; i++) cin >> w1[i];
	for (int i = 1; i <= N; i++) cin >> w2[i];

	f(1, N / 2, 0, 0, 0);
	f(N / 2 + 1, N, 0, 0, 0);
	
	for (int i = 0; i <= N / 2; i++) {
		sort(L[i].begin(), L[i].end(), [](S& a, S& b) {
			if (a.sum != b.sum) return a.sum < b.sum;
			return a.bit < b.bit;
			});
		sort(R[i].begin(), R[i].end(), [](S& a, S& b) {
			if (a.sum != b.sum) return a.sum < b.sum;
			return a.bit > b.bit;
			});
	}

	ll mn = 1e18, bit;

	for (int i = 0; i <= N / 2; i++) {
		vector<S>& LL = L[i], & RR = R[N / 2 - i];
		
		int p1 = 0, p2 = RR.size() - 1;
		while (p1 < LL.size() && p2 >= 0) {
			if (fabs(LL[p1].sum + RR[p2].sum) < mn) {
				mn = fabs(LL[p1].sum + RR[p2].sum);
				bit = LL[p1].bit + RR[p2].bit;
			}
			else if (fabs(LL[p1].sum + RR[p2].sum) == mn && LL[p1].bit + RR[p2].bit < bit) {
				mn = fabs(LL[p1].sum + RR[p2].sum);
				bit = LL[p1].bit + RR[p2].bit;
			}

			if (LL[p1].sum + RR[p2].sum <= 0) p1++;
			else p2--;
		}
	}

	for (int i = 1; i <= N; i++)
		cout << (bit & (1ll << (N - i)) ? 2 : 1) << ' ';

	return 0;
}

void f(int p, int up, int cnt, ll sum, ll bit) {
	if (p == up + 1) {
		if (up == N / 2)
			L[cnt].push_back({ sum, bit });
		if (up == N)
			R[cnt].push_back({ sum, bit });
		return;
	}

	f(p + 1, up, cnt, sum + w1[p], bit);
	f(p + 1, up, cnt + 1, sum - w2[p], bit ^ (1ll << (N - p)));
}