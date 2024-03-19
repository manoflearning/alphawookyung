#include <iostream>
#include <cstring>
#include <set>
using namespace std;

const int MAXV = 1e3;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

int R, C;
int arr[MAXV + 5][MAXV + 5];
int p[MAXV * MAXV];

inline int trans(int y, int x);
int find(int x);
void merge(int a, int b);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	cin >> R >> C;

	for (int y = 0; y < R; y++) {
		for (int x = 0; x < C; x++) {
			char n; cin >> n;
			arr[y][x] = n - '0';
		}
	}

	memset(p, -1, sizeof(p));

	for (int y = 0; y < R; y++) {
		for (int x = 0; x < C; x++) {
			if (arr[y][x] == 1) continue;
			for (int i = 0; i < 4; i++) {
				int ny = y + dy[i], nx = x + dx[i];
				if (ny < 0 || R <= ny || nx < 0 || C <= nx) continue;
				if (arr[ny][nx] == 1) continue;
				merge(trans(y, x), trans(ny, nx));
			}
		}
	}
	
	for (int y = 0; y < R; y++) {
		for (int x = 0; x < C; x++) {
			if (arr[y][x] == 0) continue;

			set<int> visited;
			for (int i = 0; i < 4; i++) {
				int ny = y + dy[i], nx = x + dx[i];
				if (ny < 0 || R <= ny || nx < 0 || C <= nx) continue;
				if (arr[ny][nx] > 0) continue;
				visited.insert(find(trans(ny, nx)));
			}

			for (auto i : visited) arr[y][x] += -p[i];
		}
	}

	for (int y = 0; y < R; y++) {
		for (int x = 0; x < C; x++) {
			cout << arr[y][x] % 10;
		}
		cout << '\n';
	}

	return 0;
}

inline int trans(int y, int x) {
	return y * MAXV + x;
}

int find(int x) {
	if (p[x] < 0) return x;
	p[x] = find(p[x]);
	return p[x];
}

void merge(int a, int b) {
	int A = find(a), B = find(b);
	if (A == B) return;
	p[A] += p[B];
	p[B] = A;
}