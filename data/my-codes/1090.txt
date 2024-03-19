#include <iostream>
#include <vector>
using namespace std;

vector<int> kmp(const string& T, const string& P);
vector<int> getpi(const string& P);
int gcd(int a, int b);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int N; cin >> N;

	string T, P;
	for (int i = 0; i < N; i++) {
		char c; cin >> c;
		P.push_back(c);
	}
	for (int i = 0; i < N; i++) {
		char c; cin >> c;
		T.push_back(c);
	}

	for (int i = 0; i < N; i++) {
		T.push_back(T[i]);
	}

	int ans = kmp(T, P).size();
	
	int GCD = gcd(ans, N);
	cout << ans / GCD << '/' << N / GCD << '\n';

	return 0;
}

vector<int> kmp(const string& T, const string& P) {
	vector<int> ret;

	vector<int> pi = getpi(P);
	
	int begin = 0, matched = 0;
	while (begin < T.size() / 2) {
		if (T[begin + matched] == P[matched]) {
			matched++;
			if (matched == P.size()) ret.push_back(begin);
		}
		else {
			if (matched == 0) begin++;
			else {
				begin += matched - pi[matched - 1];
				matched = pi[matched - 1];
			}
		}
	}

	return ret;
}

vector<int> getpi(const string& P) {
	vector<int> pi(P.size());

	int begin = 1, matched = 0;
	while (begin + matched < P.size()) {
		if (P[begin + matched] == P[matched]) {
			matched++;
			pi[begin + matched - 1] = matched;
		}
		else {
			if (matched == 0) begin++;
			else {
				begin += matched - pi[matched - 1];
				matched = pi[matched - 1];
			}
		}
	}

	return pi;
}

int gcd(int a, int b) {
	if (b == 0) return a;
	else return gcd(b, a % b);
}