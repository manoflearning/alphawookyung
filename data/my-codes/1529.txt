#include <iostream>
#include <vector>
#include <map>
#define ll long long
#define pll pair<ll, ll>
using namespace std;

const int MAX = 20000;
const ll MOD1 = 1e9 + 7, MOD2 = 1e9 + 9;

int N, K;
vector<ll> arr;
ll d, dexp1[MAX], dexp2[MAX];
map<pll, int> mp;

int bSearch(int L, int R);
bool rabinKarp(int len);
void init();

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	d = ((int)1e6 + 1);
	
	dexp1[0] = dexp2[0] = 1;
	for (int i = 1; i < MAX; i++) {
		dexp1[i] = d * dexp1[i - 1] % MOD1;
		dexp2[i] = d * dexp2[i - 1] % MOD2;
	}

	cin >> N >> K;

	arr.resize(N);
	for (auto& i : arr) cin >> i;

	cout << bSearch(0, N);

	return 0;
}

int bSearch(int L, int R) {
	while (L < R) {
		int mid = (L + R + 1) / 2;
		
		if (rabinKarp(mid)) L = mid;
		else R = mid - 1;
	}
	return L;
}

bool rabinKarp(int len) {
	init();

	ll t1 = 0, t2 = 0;
	for (int i = 0; i < len; i++) {
		t1 = (d * t1 + arr[i]) % MOD1;
		t2 = (d * t2 + arr[i]) % MOD2;
	}
	mp.insert(make_pair(make_pair(t1, t2), 1));
	
	for (int i = 1; i < N - len + 1; i++) {
		t1 = (d * (t1 - dexp1[len - 1] * arr[i - 1]) + arr[i + len - 1]) % MOD1;
		t1 = (t1 + MOD1) % MOD1;
		t2 = (d * (t2 - dexp2[len - 1] * arr[i - 1]) + arr[i + len - 1]) % MOD2;
		t2 = (t2 + MOD2) % MOD2;

		if (mp.find(make_pair(t1, t2)) != mp.end())
			mp.find(make_pair(t1, t2))->second++;
		else mp.insert(make_pair(make_pair(t1, t2), 1));
	}

	for (auto& i : mp) {
		if (i.second >= K) return true;
	}
	return false;
}

void init() {
	mp.clear();
}