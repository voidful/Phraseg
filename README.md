# Phraseg - 一言：新詞發現工具包
Tools for out of vocabulary(oov) word extraction and extracting new phrases       
通過統計的方式實現無監督新詞發現    

### Feature
- 能從極少的語料中抽取新詞
- 召回率更好
- 支持不同語言
- 輕量易用
- extract phrase from extreme small context
- better recall compare to old method
- support different language
- light and easy to use
## Usage

Install:

```
pip install phraseg
```

Before using :
```
from phraseg import *
```


# How to use
```python
phraseg = Phraseg(inputfile)
result = phraseg.extract()
```
# Function
### init(path)
Arguments  
- `inputfile(String)` : input file path


### extract()
Returns  
- `result(dict)` : result dict - oov phrase with frequency

## Example 
input file  
```text
Apple in October 2019 debuted the AirPods Pro, a new higher-end version of its existing AirPods with an updated design, noise cancellation technology, better sound, and a more expensive $249 price tag.

Apple says that with the AirPods Pro, the company is taking the magic of the AirPods "even further," with the AirPods Pro to be sold alongside the lower cost AirPods 2.

The AirPods Pro look similar to the original AirPods, but feature a wider front to accommodate silicone tips for comfort, fit, and noise cancellation purposes. Tips come in three sizes to fit different ears.

Though we heard rumors suggesting AirPods Pro might come in multiple colors, Apple is offering them only in white, much like the original AirPods.

Active Noise Cancellation is a key feature of the AirPods 2, using two microphones (one outward facing and one inward facing) along with advanced software to adapt to each ear for what Apple says is a "uniquely customized, superior noise-canceling experience."

With a built-in Transparency mode that can be toggled on, users have the option to listen to music with Active Noise Cancellation turned on while still hearing the ambient environment around them.

Inside of the AirPods Pro, there's a new vent system aimed at equalizing pressure, which Apple says will minimize the discomfort common with other in-ear designs for a better fit and a more comfortable wearing experience.
```
Result    
```text
[('the AirPods', 6),
 ('AirPods Pro', 6),
 ('the AirPods Pro', 4),
 ('Apple says', 3),
 ('of the AirPods', 3),
 ('a new', 2),
 ('noise cancellation', 2),
 ('and a more', 2),
 ('with the AirPods Pro', 2),
 ('come in', 2),
 ('Active Noise Cancellation', 2),
 ('is a', 2)]
```

input file  
```text
古文觀止

卷一‧鄭伯克段于鄢　　左傳‧隱公元年　

初，鄭武公娶于申，曰武姜，生莊公及共叔段。莊公寤生，驚姜氏，故名曰寤生，遂惡
之。愛共叔段，欲立之。亟請於武公，公弗許。

及莊公即位，為之請制。公曰：「制，巖邑也。虢叔死焉，它邑唯命。」請京，使居之
，謂之京城大叔。

祭仲曰：「都城過百雉，國之害也。先王之制，大都，不過參國之一；中，五之一；小
，九之一。今京不度，非制也，君將不堪。」公曰：「姜氏欲之，焉辟害？」對曰：「
姜氏何厭之有？不如早為之所，無使滋蔓。蔓，難圖也。蔓草猶不可除，況君之寵弟乎
？」公曰：「多行不義必自斃，子姑待之。」

既而大叔命西鄙、北鄙貳於己。公子呂曰：「國不堪貳。君將若之何？欲與大叔，臣請
事之。若弗與，則請除之，無生民心。」公曰：「無庸，將自及。」大叔又收貳以為己
邑，至于廩延。子封曰：「可矣！厚將得眾。」公曰：「不義不暱，厚將崩。」

大叔完聚，繕甲兵，具卒乘，將襲鄭；夫人將啟之。公聞其期曰：「可矣。」命子封帥
車二百乘以伐京，京叛大叔段。段入于鄢，公伐諸鄢。五月辛丑，大叔出奔共。

書曰：「鄭伯克段于鄢。」段不弟，故不言弟。如二君，故曰克。稱鄭伯，譏失教也，
謂之鄭志。不言出奔，難之也。

遂寘姜氏于城潁，而誓之曰：「不及黃泉，無相見也。」既而悔之。

潁考叔為潁谷封人，聞之。有獻於公，公賜之食，食舍肉，公問之。對曰：「小人有母
，皆嘗小人之食矣。未嘗君之羹，請以遺之。」公曰：「爾有母遺，繄我獨無。」潁考
叔曰：「敢問何謂也？」公語之故，且告之悔。對曰：「君何患焉？若闕地及泉，隧而
相見，其誰曰不然？」公從之。

公入而賦：「大隧之中，其樂也融融。」姜出而賦：「大隧之外，其樂也泄泄。」遂為
母子如初。

君子曰：「潁考叔，純孝也，愛其母，施及莊公。詩曰：『孝子不匱，永錫爾類。』其
是之謂乎！」
```
Result    
```text
[('大叔', 7), 
('莊公', 4), 
('姜氏', 4), 
('鄭伯', 3), 
('潁考', 3), 
('武公', 2), 
('及莊公', 2), 
('為之', 2), 
('謂之', 2), 
('國之', 2), 
('君將', 2), 
('不堪', 2), 
('君之', 2), 
('不義', 2), 
('既而', 2), 
('子封', 2), 
('厚將', 2), 
('出奔', 2), 
('不言', 2), 
('相見', 2), 
('潁考叔', 2), 
('之食', 2), 
('小人', 2), 
('有母', 2), 
('大隧之', 2), 
('其樂也', 2)]
```