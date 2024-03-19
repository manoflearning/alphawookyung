#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
#include <cassert>
using namespace std;
#define ll long long
#define fr first
#define sc second
#define pii pair<int, int>
#define pll pair<ll, ll>
#define vi vector<int>
#define vvi vector<vector<int>>
#define vl vector<ll>
#define vvl vector<vector<ll>>
#define vs vector<string>
#define vvs vector<vector<string>>

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1, 1, 1, -1, -1 };
const int dx[] = { 1, -1, 0, 0, 1, -1, 1, -1 };

int n;
vs s(5);
vs num[10];
vi ans[10];

void init() {
	for (int i = 0; i < 10; i++) num[i].resize(5);

	num[0] = { "###", 
			   "#.#", 
			   "#.#", 
			   "#.#", 
			   "###" };
	num[1] = { "..#", 
			   "..#", 
			   "..#", 
			   "..#", 
			   "..#" };
	num[2] = { "###", 
		       "..#", 
			   "###", 
			   "#..", 
			   "###" };
	num[3] = { "###",
			   "..#",
			   "###",
			   "..#",
			   "###" };
	num[4] = { "#.#",
			   "#.#",
			   "###",
			   "..#",
			   "..#" };
	num[5] = { "###",
			   "#..",
			   "###",
			   "..#",
			   "###" };
	num[6] = { "###",
			   "#..",
			   "###",
			   "#.#",
			   "###" };
	num[7] = { "###",
			   "..#",
			   "..#",
			   "..#",
			   "..#" };
	num[8] = { "###",
			   "#.#",
			   "###",
			   "#.#",
			   "###" };
	num[9] = { "###",
			   "#.#",
			   "###",
			   "..#",
			   "###" };
}

void input() {
	cin >> n;
	for (int i = 0; i < 5; i++) cin >> s[i];
}

void f() {
	for (int i = 0; i < 4 * n - 1; i += 4) {
		for (int j = 0; j < 10; j++) {
			int b = 1;

			for (int y = 0; y < 5; y++) {
				for (int x = 0; x < 3; x++) {
					b &= !(s[y][x + i] == '#' && num[j][y][x] == '.');
				}
			}
			if (b) ans[i / 4].push_back(j);
		}
	}
}

double f2() {
	double ret = 0;
	for (int i = 0; i < n; i++) {
		if (ans[i].empty()) return -1;

		double sum = 0;
		for (auto& j : ans[i]) sum += j;

		ret += sum / ans[i].size() * pow(10, n - i - 1);
	}
	return ret;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	init();

	input();

	f();

	cout << f2();

	return 0;
}
