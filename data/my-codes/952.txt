#include <iostream>
#include <map>
using namespace std;

const int dy[] = { 1, -1, 0, 0 };
const int dx[] = { 0, 0, 1, -1 };

char arr[10][10];
map<string, string> mp;
string s;

void dfs(int y, int x);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	for (int y = 0; y < 5; y++) {
		for (int x = 0; x < 5; x++) {
			cin >> arr[y][x];
		}
	}

	for (int y = 0; y < 5; y++) {
		for (int x = 0; x < 5; x++) {
			dfs(y, x);
		}
	}

	cout << mp.size() << '\n';

	return 0;
}

void dfs(int y, int x) {
	s.push_back(arr[y][x]);
	
	if (s.size() >= 6) {
		mp.insert({ s, s });
		s.pop_back();
		return;
	}
	
	for (int i = 0; i < 4; i++) {
		int ny = y + dy[i], nx = x + dx[i];
		if (ny < 0 || 5 <= ny || nx < 0 || 5 <= nx) continue;
		dfs(ny, nx);
	}

	s.pop_back();
}