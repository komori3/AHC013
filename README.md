#### 一日目

移動: 上下左右の最寄りが同じ色なら加点？

2回以上移動したら全域木にはできない

理論値は `K*100*99/2 = 4950*K` 

```
K=2: 9900
K=3: 14850
K=4: 19800
K=5: 24750
```

上限は 866250 点くらい？

テスターのスコア計算に高速化の余地はあるか？ → めっちゃある

出力では移動->接続の順だが、解の構成時には気にしなくてよい (接続したノードは動かせなくなる / 辺は横切れなくなる)
* 同時に最適化することも考慮に入れておく

K=5 として、サイズ 20 の連結成分を 5 個作るよりサイズ 100 の連結成分を 1 つ作ったほうが自明に良い (950 < 4950)
* 満遍なく繋ぐより、色を限定したほうがよい

`M (<K)`  色揃えることを目標とすれば、移動に `(K-M)*100` 回くらいは割くことができる

サイズ上位の連結成分を連結するのに必要な操作回数くらいは接続操作のために取っておくのがよさそう　バランス調整むずそう