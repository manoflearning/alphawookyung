#include <iostream>
using namespace std;
#define ll long long

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	ll N; cin >> N;
	
	if (N == 0) {
		cout << "NO";
		return 0;
	}

	ll flag = 1;
	while (flag <= (double)N / 3) flag *= 3;

	for (; flag >= 1; flag /= 3) {
		if (N >= flag) N -= flag;
	}
	
	if (N) cout << "NO";
	else cout << "YES";

	return 0;
}