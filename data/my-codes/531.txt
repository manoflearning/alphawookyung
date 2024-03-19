#include <iostream>
using namespace std;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int N, L;
	cin >> N >> L;

	int s = -1, len = 0;
	
	for (int i = L; i <= 100; i++) {
		if ((2 * N - i * i + i) % (2 * i) == 0 && (2 * N - i * i + i) / (2 * i) >= 0) {
			s = (2 * N - i * i + i) / (2 * i);
			len = i;
			break;
		}
	}
	
	if (s == -1) cout << -1 << '\n';
	else {
		for (int i = s; i < s + len; i++)
			cout << i << ' ';
	}

	return 0;
}