#include <iostream>
#include <vector>
using namespace std;

const int MOD = 10007;

int p[200000];
vector<int> T;
vector<int> hashTable[MOD];

int bSearch(int L, int R);
bool rabinKarp(int len);
void init();
bool match(int a, int b, int len);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	p[0] = 1;
	for (int i = 1; i < 200000; i++)
		p[i] = p[i - 1] * 26 % MOD;

	int len;
	string s;
	cin >> len >> s;

	T.resize(len);
	for (int i = 0; i < len; i++) {
		T[i] = s[i] - 'a';
	}
	
	cout << bSearch(0, len);

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

	int t = 0;
	for (int i = 0; i < len; i++) {
		t = (t + T[i] * p[len - (i + 1)]) % MOD;
	}
	hashTable[t].push_back(0);
	
	for (int i = len; i < T.size(); i++) {
		t = (26 * (t - T[i - len] * p[len - 1]) + T[i]) % MOD;
		t = (t + MOD) % MOD;
		
		for (int s : hashTable[t]) {
			if (match(s, i - len + 1, len)) return true;
		}
		hashTable[t].push_back(i - len + 1);
	}
	return false;
}

void init() {
	for (auto& i : hashTable) i.clear();
}

bool match(int a, int b, int len) {
	for (int i = 0; i < len; i++) {
		if (T[i + a] != T[i + b]) return false;
	}
	return true;
}