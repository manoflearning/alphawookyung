#include <iostream>
#include <map>
#define ll long long
#define pll pair<ll, ll>
using namespace std;

const int MAX = 2 * 1e5, dis = 333;
const ll MOD1 = 1e9 + 7, MOD2 = 1e9 + 9;

string T;
ll d = 26, dexp1[MAX], dexp2[MAX];
map<pll, int> mp[dis][dis];

int bSearch(int L, int R);
bool rabinKarp(int len);
void init();

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	dexp1[0] = dexp2[0] = 1;
	for (int i = 1; i < MAX; i++) {
		dexp1[i] = d * dexp1[i - 1] % MOD1;
		dexp2[i] = d * dexp2[i - 1] % MOD2;
	}
	
	int L; cin >> L;
	cin >> T;
	
	cout << bSearch(0, L);

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
		t1 = (d * t1 + T[i] - 'a') % MOD1;
		t2 = (d * t2 + T[i] - 'a') % MOD2;
	}
	mp[t1 % dis][t2 % dis].insert(make_pair(make_pair(t1, t2), 0));
	
	for (int i = 1; i < T.size() - len + 1; i++) {
		t1 = (d * (t1 - dexp1[len - 1] * (T[i - 1] - 'a')) + T[i + len - 1] - 'a') % MOD1;
		t1 = (t1 + MOD1) % MOD1;
		t2 = (d * (t2 - dexp2[len - 1] * (T[i - 1] - 'a')) + T[i + len - 1] - 'a') % MOD2;
		t2 = (t2 + MOD2) % MOD2;
		
		if (mp[t1 % dis][t2 % dis].find(make_pair(t1, t2)) != mp[t1 % dis][t2 % dis].end()) return true;
		else mp[t1 % dis][t2 % dis].insert(make_pair(make_pair(t1, t2), 0));
	}

	return false;
}

void init() {
	for (int i = 0; i < dis; i++) {
		for (int l = 0; l < dis; l++) {
			mp[i][l].clear();
		}
	}
}