|110|222|
|---|---|
|![fitting_image1](https://github.com/ajioka-fumito/XRD_Fitting/blob/master/README/110.png)|![fitting_image2](https://github.com/ajioka-fumito/XRD_Fitting/blob/master/README/222.png)|
# Summary
本モジュールではXRD測定によって得られたヒストグラムを２つの関数の重ねわせにより表現し,K&alpha;1に起因するヒストグラムのみを取りだすことを目的としている.  
構成要素は１つのMain関数と３つのSubクラスである.  
スペクトルフィッティングは初期値を解析解に近い値で設定する必要があるため初期値の設定に工夫が要求される.そのため本モジュール化では初期値の自動設定を含めた実装を行った.


# Main
Main関数ではデータの入出力および関数フィッティングを行う.
### \__init__
#### ka1, ka2
特性ｘ線の波長(const)
#### data, x, t
XRDの計測値(既知)    
各方位について{110theta, 110intensity, ... , 222theta, 222intensity}といったようにカラムを管理すること  
xに任意の方位に対応するtheta, tに任意の方位に対応するintensityを格納

#### x1,t_max
ヒストグラムの最大値を{t_max},最大値に対応する2&theta;を{x1}に格納  
#### delta
ka2のピークはka1のピークから少しずれることが知られている.  
実験原理より  

2d sin&theta; = n&lambda;  

以下簡単のためn=1の場合のみ考える（それが一般的らしい）  
k&alpha;1,k&alpha;2についてそれぞれ上式を用いると


2d sin(&theta;<sub>k&alpha;1</sub>) = &lambda;<sub>k&alpha;1</sub>  
2d sin(&theta;<sub>k&alpha;2</sub>) = &lambda;<sub>k&alpha;2</sub>  

ここで&theta;<sub>k&alpha;2</sub> = &theta;<sub>k&alpha;1</sub> + &Delta;&theta; より

2d sin(&theta;<sub>k&alpha;2</sub>) = 2d sin(&theta;<sub>k&alpha;1</sub> + &Delta;&theta;) = &lambda;<sub>k&alpha;2</sub>  

上式について&Delta;&theta;以外が既知であるため&Delta;&theta;は数値計算により近似値が計算可能である.  
ここで計算される&Delta;&theta;はブラック条件から算出されているためXRDの結果上では2&Delta;&theta;ずれるため2倍してから格納している
#### noise
ノイズの値を格納 (scalar)
#### init_params (vector)
2つのフィッティング関数に対して  
[noize,a<sub>k&alpha;1</sub> ,b<sub>k&alpha;1</sub>,c<sub>k&alpha;1</sub>,a<sub>k&alpha;2</sub>,b<sub>k&alpha;2</sub>,c<sub>k&alpha;2</sub>]の順に初期値を格納 
#### ・ noise
特に説明なし  
#### ・ a<sub>k&alpha;1</sub>
t_maxの0.8倍の値を設定（最大値を初期値とすると上ブレした際にそれを補うためにa<sub>k&alpha;2</sub>が負の値に飛んでしまうことがあったため）
#### ・ a<sub>k&alpha;2</sub>
[1]よりa<sub>k&alpha;1</sub>の0.4倍の値を取ると考えられているためこの値を採用  

#### ・ b<sub>k&alpha;1</sub>
x1の値を採用
#### ・ b<sub>k&alpha;2</sub>
x1 + &Delta;&theta;の値を採用
#### ・ c<sub>k&alpha;1</sub>
大体0.09~0.10の間に収まるためa<sub>k&alpha;1</sub>と同様のニュアンスで0.8に設定
#### ・ c<sub>k&alpha;2</sub>
[1]よりc<sub>k&alpha;1</sub>に対して少し大きな値を取ることが知られているため1.1倍の傾斜をかけた  
[2]では２つのピークが相似形で有ることを仮定して議論をすすめていたため,傾斜なしでも試したが,怪しい挙動を示したためやめておくのが無難.一応この現象を裏付ける文献[a3]も見つけておいた  

#### popt
フィッティング後のパラメータを格納. 構成は init_params と同様
# Subclass
Subclassは3つ存在していて用途別に分類した.
## Visualize
### plot
fittingが適切に行われたかを確認
## SubFunction
### bragg
bragg条件に基づく関数で,以下delta関数で用いる.  
numpyの三角関数計算はrad単位で行われるが,XRDのデータはdegree単位であるため変換を挟んでいる.  
**出力:deltaを代入したときの絶対値誤差(scalar)**
### max_intensity
初期値の設定に測定結果の最大値と対応する角度が必要となる  
**出力:{対応するx,tの最大値}(vector)**
### delta
&Delta;&theta;の近似解を計算する.  
&Delta;&theta;の値が高々0.5(degree)であること,XRDの測定幅から有効数字が下3桁であることを考えてこの実装にした.  
**出力:deltaの計算解(scalar)**
### noise
ヒストグラムのビークは中心に位置する.そのためヒストグラムの前後100点の平均値をノイズの初期値とした.  
**出力:noise(scalar)**
## FittingFunction
curve fitting を行う際に使用する関数群
### ・gauss
入力に対応するガウス関数のyを返す  
**引数：ｘ(vector), params(a,b,c)**  
**返り値：入力したｘ(vector)に対応するy(vector)**
### ・gauss_puls
ka1とka2に対応するガウス関数を重ね合わせた結果を出力する関数  
**引数：ｘ(vector), params(a,b,c)**  
**返り値：入力したｘ(vector)に対応するpredict(vector)**

>[1] http://www.crl.nitech.ac.jp/~ida/research/preprints/how_to_use_pxrd_5.pdf  
>[2] W. A. Rachinger, J. Sci. Instrum., 1948, 25, 254.  
>[3] https://www.researchgate.net/profile/Takashi_Ida/publication/200045553_Deconvolution_of_the_instrumental_functions_in_powder_X-ray_diffractometry/links/546c08070cf20dedafd53b9a.pdf