# Phraseg - 一言：新詞發現工具包
Tools for out of vocabulary(oov) word extraction and extracting new phrases without supervision         
通過統計的方式實現無監督新詞發現    

### Feature
- 能從極少的語料中抽取新詞
- 召回率更好
- 支持不同語言
- 輕量易用
- extract phrases on very little content
- better recall compare to old method
- support different language
- light and easy to use

### How it works?
English: 
[https://medium.com/@voidful.stack/simple-and-effective-phrase-finding-in-multi-language-42264554acb](https://medium.com/@voidful.stack/simple-and-effective-phrase-finding-in-multi-language-42264554acb)  

Chinese: 
[https://voidful.github.io/voidful_blog/implement/2019/09/03/oov-detection-implement/](https://voidful.github.io/voidful_blog/implement/2019/09/03/oov-detection-implement/)  


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
Arguments  
- `filter(bool)` :  filter not likely result  
Returns  
- `result(dict)` : result dict - oov phrase with frequency

### extract_sent()   
Arguments  
- `sentence(String)` : get phrase from sentence  
- `filter(bool)` : filter not likely result    

Returns  
- `result(dict)` : result dict - oov phrase with frequency


### Colab Demo  
[https://colab.research.google.com/drive/1n-JVX7XPupWz3RuoOOMv-1sQAhMo3sQo](https://colab.research.google.com/drive/1n-JVX7XPupWz3RuoOOMv-1sQAhMo3sQo)


## Example  
### English
input
```text
The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU [16], ByteNet [18] and ConvS2S [9], all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions. In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions [12]. In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2.
Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations [4, 27, 28, 22].
End-to-end memory networks are based on a recurrent attention mechanism instead of sequence- aligned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks [34].
To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence- aligned RNNs or convolution. In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as [17, 18] and [9].
```
Result    
```text
[('the Transformer', 3), ('ConvS 2 S', 2), ('input and output', 2), ('output positions', 2), ('number of operations', 2), ('In the', 2), ('attention mechanism', 2), ('to compute', 2)]
```

### Chinese
input  
```text
Wiki（聆聽i/ˈwɪkiː/）是在全球資訊網上開放，且可供多人協同創作的超文字系統，由沃德·坎寧安於1995年首先開發。沃德·坎寧安將wiki定義為「一種允許一群用戶用簡單的描述來建立和連接一組網頁的社會計算系統」[1]。
有些人認為[2]，Wiki系統屬於一種人類知識的網路系統，讓人們可以在web的基礎上對Wiki文字進行瀏覽、建立和更改，而且這種建立、更改及發布的成本遠比HTML文字小。與此同時，Wiki系統還支援那些面向社群的協作式寫作，為協作式寫作提供了必要的幫助。最後Wiki的寫作者自然構成了一個社群，Wiki系統為這個社群提供了簡單的交流工具。與其它超文字系統相比，Wiki有使用簡便且開放的特點，有助於在一個社群內共享某個領域的知識。
```
Result    
```text
[('系統', 7), ('文字', 4), ('社群', 4), ('建立', 3), ('Wiki系統', 3), ('個社群', 3), ('開放', 2), ('超文字系統', 2), ('沃德坎寧安', 2), ('一種', 2), ('用簡', 2), ('知識', 2), ('更改', 2), ('協作式寫作', 2), ('提供了', 2)]
```

### Japanese
input
```text
作品の概要
本作は、22世紀の未来からやってきたネコ型ロボット・ドラえもんと、勉強もスポーツも苦手な小学生・野比のび太が繰り広げる少し不思議（SF）な日常生活を描いた作品である。基本的には一話完結型の連載漫画であるが、一方でストーリー漫画形式となって日常生活を離れた冒険をするという映画版の原作でもある「大長編」シリーズもある。一話完結の基本的なプロットは、「ドラえもんがポケットから出す多種多様なひみつ道具（現代の技術では実現不可能な機能を持つ）で、のび太（以外の場合もある）の身にふりかかった災難を一時的に解決するが、道具を不適切に使い続けた結果、しっぺ返しを受ける」というものが多く、前作の「ウメ星デンカ」のストーリー構図をほぼそのまま踏襲しており実質的な後継作品ともいえる。このプロットは、作者の藤子・F・不二雄が自身のSF作品で描いた独自定義「すこし・不思議」（Sukoshi Fushigi）[注 2]という作風に由来し、当時の一般SF作品の唱える「if」（もしも） についての対象を想定した回答が反映されている。
作品の主人公はドラえもんであるが、上記のプロットのように物語の主な視点人物はのび太である。
```
Result    
```text
[('ある', 7), ('作品', 6), ('ット', 5), ('である', 4), ('ドラえもん', 3), ('ラえもん', 3), ('のび太', 3), ('び太', 3), ('という', 3), ('いう', 3), ('プロット', 3), ('から', 2), ('って', 2), ('日常生活を', 2), ('描いた', 2), ('基本的', 2), ('的に', 2), ('一話完結', 2), ('漫画', 2), ('ストーリー', 2), ('する', 2), ('的な', 2), ('道具', 2), ('SF作品', 2), ('の主', 2)]
```

### Cantonese
input
```text
「硬膠」原本是指硬身的塑膠，所以亦有「軟膠」，但在網上有人用此詞取其諧音代替戇鳩這個粵語粗口；直譯英語為 Hard Plastic，繼而引申出縮寫 HP。此用詞是諧音粗口，帶有不雅或恥笑成份。部分人不接受說「膠」字，認為膠是粗口之一[1]。
硬膠亦簡稱「膠」，可作名詞、形容詞、動詞等使用。硬膠雖與「on9」的語調上不同，但意思差不多，有時可以相通。小丑icon和小丑神也是硬膠的形象化的圖像顯示和象徵。
「硬膠」一詞聽聞歷史悠久，但出處不明，而由香港高登討論區將其發揚光大，現己推展至香港其他主要網絡社區。在2002年時，有些網民利用「戇鳩」的諧音，發明「硬膠」一詞，更把愛發表無厘頭帖子的會員腦魔二世定為「硬膠」始祖。自此，「硬膠文化」便慢慢發展起來，某程度上硬膠文化與無厘頭文化相似。因腦魔二世的「硬膠」功力驚人，更成為當時被剝削的會員的救星，可令他們苦中作樂一番，故有人曾經預言：「救高登，靠膠人」。及後，高登會員以縮寫「膠」來代替硬膠。而高登亦有了「膠登」的綽號。當時甚至出現了7位膠力驚人的高登會員，以腦魔二世為首，合稱為「硬膠七子」。
其實「硬膠」早於周星馳電影《整蠱專家》早已出現雛型。戲中有一台詞為「超級戇膠膠」，可見膠字是用作取代粵語粗口鳩字。
有網友提供資料指西方早於上世紀六十年代亦已經將「膠」（Plastics）等同愚蠢
```
Result    
```text
[('硬膠', 13), ('高登', 5), ('粗口', 4), ('諧音', 3), ('腦魔二世', 3), ('魔二世', 3), ('亦有', 2), ('代替', 2), ('戇鳩', 2), ('粵語粗口', 2), ('縮寫', 2), ('小丑', 2), ('一詞', 2), ('香港', 2), ('無厘頭', 2), ('的會員', 2), ('力驚人', 2), ('當時', 2), ('出現', 2), ('早於', 2)]
```

### Korean
input
```text
"서울"의 어원에 대해서는 여러 가지 설이 존재하나, 학계에서는 일반적으로 수도(首都)를 뜻하는 신라 계통의 고유어인 서라벌에서 유래했다는 설이 유력하게 받아들이고 있다. 이때 한자 가차 표기인 서라벌이 원래 어떤 의미였을지에 대해서도 여러 학설이 존재한다. 삼국사기 등에서 서라벌을 금성(金城)으로도 표기했다는 점과 신라(新羅)까지 포함하여 "설[새: 新, 金]-벌[땅: 羅, 城]", 즉 '새로운 땅'이라는 뜻으로 새기는 견해가 있다. 고대-중세 한국어에서 서라벌을 정확히 어떻게 발음했는지에 대해서는 확실하게 확인된 바가 없으며, 그 발음을 알 수 있게 되는 것은 훈민정음의 창제 후 "셔ᄫᅳᆯ"이라는 표기가 등장하고 나서부터이다.
조선 시대에는 서울을 한양 이외에도 경도(京都), 경부(京府), 경사(京師), 경성(京城), 경조(京兆) 등으로 쓰는 경우가 종종 있었으며, 김정호의 수선전도에서 알 수 있듯 수선(首善)으로 표기한 예도 있다. 그 밖의 표기 중에는 서울의 한자 음차 표기로서 박제가가 북학의에서 썼던 '徐蔚(서울)'이 있다. 이는 모두 수도를 뜻하는 일반명사들로서 '서울'이 원래는 서울 지역(사대문 안과 강북의 성저십리)을 가리키는 말이 아닌 수도를 뜻하는 일반명사였다는 방증이다. 국어사전에서는 일반명사 '서울'을 '한 나라의 중앙 정부가 있고, 경제, 문화, 정치 등에서 가장 중심이 되는 도시'라고 정의하고 있다.[4] 1910년 10월 1일에 일제가 한성부를 경성부(京城府)로 개칭하면서 일제강점기에 서울은 주로 경성(京城, 일본어로는 けいじょう)으로 불렸으며, 1945년 광복 후에는 '경성'이란 말은 도태되고 거의 '서울'로 부르게 되었다.[
```
Result    
```text
[('서울', 9), ('으로', 6), ('표기', 6), ('경성', 4), ('설이', 3), ('다는', 3), ('京城', 3), ('여러', 2), ('신라', 2), ('인서라벌', 2), ('했다는', 2), ('한자', 2), ('차표기', 2), ('지에대해서', 2), ('등에서', 2), ('국어', 2), ('발음', 2), ('알수있', 2), ('하고', 2), ('수선', 2), ('성부', 2)]
```

### French
input
```text
El idioma francés (le français /lə fʁɑ̃sɛ/ ( escuchar) o la langue française /la lɑ̃ɡ fʁɑ̃sɛz/) es una lengua romance hablada en la Unión Europea, especialmente en Francia, país en el que se habla junto con otras lenguas regionales como el idioma bretón (Bretaña), el occitano (Occitania), el vasco (país vasco francés), el catalán (Rosellón), y el corso (Córcega). En los territorios franceses de ultramar es hablado en muchos casos junto con otras lenguas como el tahitiano (Polinesia Francesa), o el créole (isla Reunión, Guadalupe y Martinica). También se habla en Canadá, Estados Unidos (francés cajún, créole y francés acadio o acadiano), Haití (con el créole), y numerosos países del mundo. Según estimaciones de la Organización Internacional de la Francofonía (basadas en proyecciones demográficas de las Naciones Unidas), en el transcurso del s. XXI, el francés se convertiría en el tercer idioma con el mayor número de hablantes del mundo, sobre todo por el crecimiento poblacional de los países africanos francófonos.5​
```
Result    
```text
[('francés', 5), ('és', 5), ('ón', 5), ('paí', 4), ('en el', 3), ('créole', 3), ('éole', 3), ('franç', 2), ('fʁɑsɛ', 2), ('se habla', 2), ('junto con otras lenguas', 2), ('como el', 2), ('ún', 2), ('de la', 2), ('ía', 2)]
```

### Spanish
input
```text
Breaking Bad es una serie de televisión dramática estadounidense creada y producida por Vince Gilligan. Breaking Bad narra la historia de Walter White (Bryan Cranston), un profesor de química con problemas económicos a quien le diagnostican un cáncer de pulmón inoperable. Para pagar su tratamiento y asegurar el futuro económico de su familia comienza a cocinar y vender metanfetamina,1​ junto con Jesse Pinkman (Aaron Paul), un antiguo alumno suyo. La serie, ambientada y producida en Albuquerque (Nuevo México), se caracteriza por poner a sus personajes en situaciones que aparentemente no tienen salida, lo que llevó a que su creador la describa como un wéstern contemporáneo.2​
La serie se estrenó el 20 de enero de 2008 y es una producción de Sony Pictures Television. En Estados Unidos y Canadá se emitió por la cadena AMC.3​ La temporada final se dividió en dos partes de ocho episodios cada una y se emitió en el transcurso de dos años: la primera mitad se estrenó el 15 de julio de 2012 y concluyó el 2 de septiembre de 2012, mientras que la segunda mitad se estrenó el 11 de agosto de 2013 y concluyó el 29 de septiembre del mismo año.    
```
Result    
```text
[('ón', 3), ('seestrenóel', 3), ('estrenóel', 3), ('Breaking Bad', 2), ('es una', 2), ('y producida', 2), ('econó', 2), ('La serie', 2), ('seemitió', 2), ('emitió', 2), ('óen', 2), ('añ', 2), ('mitadseestrenóel', 2), ('de 2012', 2), ('yconcluyóel', 2), ('concluyóel', 2), ('de septiembre', 2)]
```

### Thai
input
```text
ิตฮับ (อังกฤษ: GitHub) เป็นเว็บบริการพื้นที่ทางอินเทอร์เน็ต (hosting service) สำหรับเก็บการควบคุมการปรับปรุงแก้ไข (version control) โดยใช้กิต (Git) โดยมากจะใช้จัดเก็บรหัสต้นฉบับ (source code) แต่ยังคงคุณสมบัติเดิมของกิตไว้ อย่างการให้สิทธิ์ควบคุมและปรับปรุงแก้ไข (distributed version control) และระบบการจัดการรหัสต้นฉบับรวมถึงทางกิตฮับได้เพิ่มเติมคุณสมบัติอื่นๆผนวกไว้ด้วย เช่น การควบคุมการเข้าถึงรหัสต้นฉบับ (access control) และ คุณสมบัติด้านความร่วมมือเช่น ติดตามข้อบกพร่อง (bug tracking), การร้องขอให้เพิ่มคุณสมบัติอื่นๆ (feature requests), ระบบจัดการงาน (task management) และวิกิสำหรับทุกโครงการ[2]
```
Result    
```text
[('ปร', 4), ('และ', 4), ('ละ', 4), ('กไ', 3), ('จด', 3), ('ขอ', 3), ('ทาง', 2), ('โดย', 2), ('ถง', 2)]
```