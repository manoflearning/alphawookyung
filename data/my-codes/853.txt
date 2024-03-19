#include <iostream>
#include <vector>
#include <map>
using namespace std;
#define ll long long

int T, n, m;
vector<ll> A, B, psumA, psumB;
map<ll, int> mp;

void input();

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	input();

	psumA.resize(n + 1);
	psumB.resize(m + 1);

	for (int i = 1; i < A.size(); i++) {
		psumA[i] = psumA[i - 1] + A[i];
	}

	for (int i = 1; i < B.size(); i++) {
		psumB[i] = psumB[i - 1] + B[i];
	}

	for (int i = 0; i < A.size(); i++) {
		for (int l = i + 1; l < A.size(); l++) {
			map<ll, int>::iterator it = mp.find(psumA[l] - psumA[i]);

			if (it != mp.end()) it->second++;
			else mp.insert({ psumA[l] - psumA[i], 1 });
		}
	}

	ll cnt = 0;

	for (int i = 0; i < B.size(); i++) {
		for (int l = i + 1; l < B.size(); l++) {
			ll left = T - (psumB[l] - psumB[i]);

			map<ll, int>::iterator it = mp.find(left);
			if (it != mp.end()) cnt += it->second;
		}
	}

	cout << cnt;

	return 0;
}

void input() {
	cin >> T >> n;

	A.resize(n + 1);
	for (int i = 1; i < A.size(); i++) {
		cin >> A[i];
	}

	cin >> m;

	B.resize(m + 1);
	for (int i = 1; i < B.size(); i++) {
		cin >> B[i];
	}
}