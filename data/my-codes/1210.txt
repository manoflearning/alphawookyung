#include <iostream>
using namespace std;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int N, M;
	cin >> N >> M;

	for (int i = 0; i < N; i++) {
		for (int l = 0; l < M; l++) {
			cout << i * M + (l + 1);
			if (l < M - 1) cout << ' ';
		}
		cout << '\n';
	}

	return 0;
}