class Solution {
public:
    bool isVowel(char c) {
        if (c == 'a') return 1;
        if (c == 'e') return 1;
        if (c == 'i') return 1;
        if (c == 'o') return 1;
        if (c == 'u') return 1;
        return 0;
    }
    int vowelStrings(vector<string>& words, int left, int right) {
        int ret = 0;
        for (int i = left; i <= right; i++) {
            if (!isVowel(words[i][0])) continue;
            if (!isVowel(words[i].back())) continue;
            ret++;
        }
        return ret;
    }
};