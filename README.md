# physics-seminar-2021-poster
2021年度物理工学特別輪講@小芦研 ポスター発表
## qulacs
qulacsはpartial_traceが使えるバージョンでなくてはいけない

## simulate.py
以下の形式のjsonファイルのパスを引数にとる
```json
{
    "title": "Haar",
    "n": 9,
    "k": 2,
    "r": 10,
    "type": "haar",
    "depth": 1000,
    "coupling_constant": []
}
```
時刻によって定まるprefixのついたファイルを４つ生成する

## plot.py
plot.txtにprefixを改行区切りで書くとそれに対応するデータを一つのグラフにプロットする。L1 normか相互情報量かcoherent informationかは手動で変えてください...

