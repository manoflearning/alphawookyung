#include <iostream>
#include <vector>
using namespace std;

bool kmp(const string& T, const string& P);
vector<int> getpi(const string& P);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	string T1, T2, P;
	cin >> T1 >> T2 >> P;

	if (kmp(T1, P) && kmp(T2, P)) cout << "YES\n";
	else cout << "NO\n";

	return 0;
}

bool kmp(const string& T, const string& P) {
	vector<int> pi = getpi(P);

	int begin = 0, matched = 0;
	while (begin + matched < T.size()) {
		if (matched < P.size() && T[begin + matched] == P[matched]) {
			matched++;
			if (matched == P.size()) return true;
		}
		else {
			if (matched == 0) begin++;
			else {
				begin += matched - pi[matched - 1];
				matched = pi[matched - 1];
			}
		}
	}

	return false;
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