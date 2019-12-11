import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

import unittest
from phraseg import *


class Test(unittest.TestCase):

    def testFile(self):
        phraseg = Phraseg("./smailltext")
        result = phraseg.extract()
        print(result)
        self.assertTrue(len(phraseg.ngrams) > 1)
        self.assertTrue(len(result) > 1)

    def testEng(self):
        phraseg = Phraseg('''
Apple in October 2019 debuted the AirPods Pro, a new higher-end version of its existing AirPods with an updated design, noise cancellation technology, better sound, and a more expensive $249 price tag.

Apple says that with the AirPods Pro, the company is taking the magic of the AirPods "even further," with the AirPods Pro to be sold alongside the lower cost AirPods 2.

The AirPods Pro look similar to the original AirPods, but feature a wider front to accommodate silicone tips for comfort, fit, and noise cancellation purposes. Tips come in three sizes to fit different ears.

Though we heard rumors suggesting AirPods Pro might come in multiple colors, Apple is offering them only in white, much like the original AirPods.

Active Noise Cancellation is a key feature of the AirPods 2, using two microphones (one outward facing and one inward facing) along with advanced software to adapt to each ear for what Apple says is a "uniquely customized, superior noise-canceling experience."

With a built-in Transparency mode that can be toggled on, users have the option to listen to music with Active Noise Cancellation turned on while still hearing the ambient environment around them.

Inside of the AirPods Pro, there's a new vent system aimed at equalizing pressure, which Apple says will minimize the discomfort common with other in-ear designs for a better fit and a more comfortable wearing experience.
           ''')
        result = phraseg.extract()
        print(result)
        self.assertTrue(len(phraseg.ngrams) > 1)
        self.assertTrue(len(result) > 1)

    def testChi(self):
        phraseg = Phraseg('''
        卷一‧周鄭交質　　左傳‧隱公三年

        鄭武公、莊公為平王卿士，王貳于虢，鄭伯怨王。王曰：「無之。」故周鄭交質。王子
        狐為質於鄭，鄭公子忽為質於周。
        
        王崩，周人將畀虢公政。四月，鄭祭足帥師取溫之麥；秋，又取成周之禾。周鄭交惡。
        
        君子曰：「信不由中，質無益也。明恕而行，要之以禮，雖無有質，誰能間之？苟有明
        信，澗溪沼沚之毛，蘋蘩薀藻之菜，筐筥錡釜之器，潢汙行潦之水，可薦於鬼神，可羞
        於王公。而況君子結二國之信，行之以禮，又焉用質？風有采蘩、采蘋，雅有行葦、泂
        酌，昭忠信也。」
        ''')
        result = phraseg.extract()
        print(result)
        self.assertTrue(len(phraseg.ngrams) > 1)
        self.assertTrue(len(result) > 1)

    def testChiLong(self):
        phraseg = Phraseg('''
        好看 真的好看

一種十年磨一劍的感覺

基本上是三本柱 美國隊長 鋼鐵人 雷神索爾的電影 XD

一開始鷹眼開開心心的教女兒射箭 就知道沒五分鐘就要剩粉了QQ

接著是鋼鐵人的部分 雖然跟預告一樣 也知道他不會就掛在太空船

只是還是擔心 直到驚奇隊長閃閃發光的把鋼鐵人閃醒XD

飛回地球 鋼鐵人與美隊的小爭執 我們真的一起輸了 有種揪心感

就算後來飛去找薩諾斯 再次上演手套爭奪戰

索爾這次飛快的砍掉手(復3沒人有刀來著XD

手套掉下來 只是沒了寶石 沒了寶石 沒了寶石 

當下跟著影中人一起錯愕了一下

薩老緩緩說出我用寶石滅了寶石

涅布拉還默默補充我爸不會說謊...

雷神很雷的就把薩諾斯的頭給剁了 想問甚麼也沒得問了= =(雖然是有呼應復3拉

接著就進入哀傷的五年後

美隊在開導大家

黑寡婦在當代理局長

驚奇隊長又飛去宇宙了 所以只回來打了一架

發現沒辦法逆轉現況之後就又飛走了 說好的逆轉無限呢-.-

然後美隊跟黑寡婦聊天 接著被老鼠救出來的蟻人跑來

開啟了這集最重要的量子空間回過去

然後去找了幸福美滿的東尼QQ

可以理解東尼不想幫忙 因為他的模型還沒做出來(誤

只是看著小蜘蛛的照片 讓他又爬起來努力的跑量子模型

另一邊美隊去找了可以維持浩克先生的班納

然後各種搞笑XD

蟻人各種變 有點擔心他回不來 哈哈

這時 東尼帶著金頭腦跟盾牌回來啦~~~

隊長還是要拿著圓盾比較有FU

火箭跟浩克先生去找了雷神 喔 是肥神索爾回來

索爾真的很懊悔 從媽媽救不到 爸爸在眼前消失 弟弟被薩諾斯掛了

朋友因為自己沒有給薩諾斯致命一擊而灰飛煙滅了一半

五年來不斷酗酒走不出來也是蠻合理的

只是這集三本柱最雷的就是他了XD

然後黑寡婦去找到了變成浪人的鷹眼 把他帶回來

回到總部 分隊完

穿好量子戰衣

出發

美國隊長 + 鋼鐵人 + 蟻人 + 浩克先生-> 復1的紐約

火箭浣熊 + 肥神 -> 雷神2的阿斯嘉

涅布拉 + 戰爭機器 + 鷹眼 + 黑寡婦 -> 星際異工隊1

總覺得最後一個分組怪怪的 怎麼會給三個人類去外星球搶東西= =





復1的紐約

浩克看到自己以前的樣子 學著砸了一些東西的場景真的很逗趣XD

然後他去找當時守護寶石的古一 開始了一串辯論

浩克先生這集真的沒甚麼武戲 都在講話

然後隊長走進電梯 場景跟美2一模一樣

本來以為要開扁了 結果居然正直的美隊以一句Hail Hydra輕鬆A走了權杖

順便致敬漫畫 XD

也順勢解釋為什麼復1之後權杖又跑去九頭蛇那 做實驗做出了緋紅女巫&快銀

鋼鐵人&蟻人 本來要搶到空間魔方 結果來一個浩克討厭走樓梯XD

寶石就這樣被洛基A走順便烙跑

然後隊長走著走著遇到了自己

以為未來隊長是洛基的過去隊長 開始對A

然後又來一句隊長經典台詞

I can do this all day , 我知道XD

終究是自己了解自己 以一句巴奇還活著影響自己 順利尻飛過去隊長

然後權杖戳下去 帶走權杖

會合後再度前往1970 去找還在神盾局的魔方&皮姆

這邊偷東西就沒甚麼意外 再有意外就演不完了XD

這邊的主軸 給了鋼鐵人與爸爸的相會

為人父的鋼鐵人有太多事想跟爸爸講 終於圓夢

以及隊長看卡特 為結局埋了伏筆

原來東尼本來不叫東尼XD

阿斯嘉這邊

肥神各種雷 回到過去還在找酒

然後看到媽媽就忍不住想要回去找媽媽

完全不顧大局 單就這個行為真的很雷

只是這集就是給三本柱圓夢的 當然要給肥神回去找媽媽聊天

搶寶石就交給火箭 他也很輕鬆地搶到了寶石

然後圓完雷神的夢

雷神把喵喵槌叫了過來 帶回未來(當下OS是他把槌子拿走了 待會過去的雷神要怎麼用啊

不過後面就知道雷神回到過去最重要的其實就是把槌子拿走 哈哈哈哈





星際異攻隊這邊

鷹眼跟黑寡婦搭著太空船飛到了佛米爾星

話說 哪來的太空船

而且給兩個人類四處太空旅行 每個星球都有氧氣來著= =??

然後又遇到了守門員紅骷顱

接著開始誰要跳 鷹眼你射箭阻止黑寡婦的時候壓根就想炸死他吧XDDD

然後兩個人一起跳下去的時候

我以為會出現老梗之 互相珍惜所愛的人 並且能為對方而死

才是得到靈魂寶石的正確方法

結果是出現另一個老梗 下面的人鬆手往下跳

黑寡婦真的死了!? 靈魂寶石GET (當下想說不是有個人電影 後來才知道是前傳

涅布拉跟戰爭機器

一拳KO星爵就拿到寶石了

只是本篇最雷的涅布拉不小心跟過去涅布拉連結在一起

然後就被抓了 各種劇透給2014薩諾斯

知道復仇者在過去偷寶石之後 決定跟著到未來

因為寶石已經有人幫忙收集好了

只是 皮姆粒子不是只有一個人來回

後來是怎麼讓2014涅布拉去2023 又可以開啟量子通道給薩諾斯大軍過來呢??

畫面回到2023 過去不管多久 未來其實只過了一秒

所以也能解釋涅布拉被換了也不知道 因為未來沒有比較晚回來



接著把寶石裝上手套 (說好的手套要由矮人做才能發揮威力勒~~~

關在小房間 準備彈指

為啥不把捏布拉也關進來 放他在外面開量子通道 嘖嘖

順利彈指 浩克先生差點就掛了

外面鳥開始飛 鷹眼的手機響了

蟻人開始開心的說我們做到了!

只是看電影的大家都知道 糟了

接著總部就被各種轟炸

薩諾斯坐在外面等著倖存的復仇者出來

然後 三本柱 帥氣登場

開始三英戰呂布

只是薩諾斯就算沒手套也是很強

肥神應該是因為變太肥了 感覺很弱

還被壓著差點被自己的風暴毀滅者戳死

這時

隊長

舉起喵喵槌拉!!!!!!!

超帥 不解釋 看得我都泛淚了

然後帥打一波 還有槌子+盾牌合體技

只是帥打一波之後又被屌虐一波回來

連盾牌都被打飛一半

還被嘴+大軍壓境

隊長渾身傷

只默默拉緊盾牌的帶子

再次地站了起來

雖千萬人吾往矣阿!!!!

超帥的 帥到我又泛淚了

這時絕望感來到最高

一句 Cap do you copy! On your left

然後火圈開始冒出

終於

全部人都回來了

Avengers assemble

接著開始各種秀

鋼鐵小辣椒 女武神天馬 黃蜂女 黑豹 奇異博士 小蜘蛛

手套接力賽其實只是秀大家技能的時候

隊長跟索爾交換武器 索爾還說你拿小的就好

葛魔拉怒踢星爵蛋蛋 涅布拉說你只有他跟樹可以選

然後緋紅女巫屌打2014薩諾斯 只是2014薩諾斯表示你誰啊XD

2014薩諾斯表示: 我啥事都沒做 干我屁事XD

被女巫打得受不了了 惱羞開戰艦砲轟地面

還在想這台煩人的戰艦怎麼辦 就有道紅光來解決一切

驚奇隊長終於飛回來了- -

都快打完了還不回來

打飛戰艦後

給驚奇隊長跟薩諾斯單挑一波

寶石原來還可以單拿起來轟人 只能說薩諾斯真的很會用寶石

接著就是東尼看了正在治水的奇異博士

博士只默默伸出食指

然後東尼就衝去搶手套了

本來以為手套又搶失敗了

薩諾斯拉好手套 這次沒人阻止他了

薩諾斯: I'm Inevitable

!? 啥事都沒發生

只見東尼舉起手 寶石已經全都A過來了











                           東尼 : I am Iron Man











一彈指 一切就結束了

只是鋼鐵人 就這樣華麗的謝幕了

其實繼續打下去 只要手套不被薩諾斯拿走

感覺復仇者方應該還是能贏

應該只是要給鋼鐵人一個豪華的便當才這樣的吧?

後面就是各英雄回歸自己圓滿的結局 (東尼QQ

然後隊長要歸還無限寶石 以免其他時空混亂

54321之後 想不到居然沒回來

看起來只有巴奇不緊張 然後回頭一個老人

是回到過去過平靜生活的隊長 然後將盾牌與美國隊長 交接給了獵鷹

隊長的那支舞 終於跳到了

三本柱之所以如此令人感動

終究是漫威十年來各個電影細膩的刻畫這三個腳色

美國隊長 - 美國隊長123  復仇者1234

鋼鐵人 - 鋼鐵人123 復仇者1234

雷神 - 雷神123 復仇者1234

三個人都用了七部電影 細細刻畫著這三個腳色

美隊從過去 到現在 從對抗納粹 到現在對抗不同的外星人 堅持著自己的正義

鋼鐵人從一個商人 變成了保衛地球 宇宙的超級英雄 堅持著守護自己與所愛的人

雷神從阿斯嘉的王子 接著失去一切 又重新站起 到最後重新出發

感謝漫威十年如此精采的電影



最後 終究有幾個小小問題 不知道是Bug還是我沒注意 不知道有沒有人看到的

1. 皮姆粒子與量子衣

   2019涅布拉其實只有一組來回的皮姆粒子

   2014涅布拉把皮姆粒子給薩諾斯後 應該就沒辦法回到2019?

   就像美隊跟鐵人 要飛去1970的意思一樣 因為那裏不只有空間魔方 還有皮姆粒子

   而且薩諾斯大軍也都沒有量子衣 這樣到底怎麼飛到未來的= =? (太空船比較高級XD?

   而且回到過去的飛去佛米爾星的太空船又是哪來的?

2. 無限手套

   鋼鐵人很輕鬆地就做出了無限手套 然後把寶石放上去

   浩克博士還嚇了他一下

   復3不是說只有矮人做的無限手套才能發揮寶石的能力嗎?

3. A寶石

   最後東尼A走寶石 應該是因為他手套上有動手腳來著?

   也太好拔了 應該也只是為了華麗便當作準備

4. 美國隊長的圓盾

   不是在大戰時被打壞了一半 怎麼傳給獵鷹的時候又好了?

   雖然我覺得美隊只是純粹圓夢 應該沒有要解釋這段的合理性了

突然想到回來補充

覺得4的薩諾斯相較比較扁平跟壞人

不像3的有自己的理想跟抱負

應該是因為要塑造一個打爆他也不可惜的魔王吧


--
        ''')
        result = phraseg.extract()
        print(result)
        self.assertTrue(len(phraseg.ngrams) > 1)
        self.assertTrue(len(result) > 1)

    def testOneSent(self):
        phraseg = Phraseg('''
        好看 真的好看

一種十年磨一劍的感覺

基本上是三本柱 美國隊長 鋼鐵人 雷神索爾的電影 XD

一開始鷹眼開開心心的教女兒射箭 就知道沒五分鐘就要剩粉了QQ

接著是鋼鐵人的部分 雖然跟預告一樣 也知道他不會就掛在太空船

只是還是擔心 直到驚奇隊長閃閃發光的把鋼鐵人閃醒XD

飛回地球 鋼鐵人與美隊的小爭執 我們真的一起輸了 有種揪心感

就算後來飛去找薩諾斯 再次上演手套爭奪戰

索爾這次飛快的砍掉手(復3沒人有刀來著XD

手套掉下來 只是沒了寶石 沒了寶石 沒了寶石 

當下跟著影中人一起錯愕了一下

薩老緩緩說出我用寶石滅了寶石

涅布拉還默默補充我爸不會說謊...

雷神很雷的就把薩諾斯的頭給剁了 想問甚麼也沒得問了= =(雖然是有呼應復3拉

接著就進入哀傷的五年後

美隊在開導大家

黑寡婦在當代理局長

驚奇隊長又飛去宇宙了 所以只回來打了一架

發現沒辦法逆轉現況之後就又飛走了 說好的逆轉無限呢-.-

然後美隊跟黑寡婦聊天 接著被老鼠救出來的蟻人跑來

開啟了這集最重要的量子空間回過去

然後去找了幸福美滿的東尼QQ

可以理解東尼不想幫忙 因為他的模型還沒做出來(誤

只是看著小蜘蛛的照片 讓他又爬起來努力的跑量子模型

另一邊美隊去找了可以維持浩克先生的班納

然後各種搞笑XD

蟻人各種變 有點擔心他回不來 哈哈

這時 東尼帶著金頭腦跟盾牌回來啦~~~

隊長還是要拿著圓盾比較有FU

火箭跟浩克先生去找了雷神 喔 是肥神索爾回來

索爾真的很懊悔 從媽媽救不到 爸爸在眼前消失 弟弟被薩諾斯掛了

朋友因為自己沒有給薩諾斯致命一擊而灰飛煙滅了一半

五年來不斷酗酒走不出來也是蠻合理的

只是這集三本柱最雷的就是他了XD

然後黑寡婦去找到了變成浪人的鷹眼 把他帶回來

回到總部 分隊完

穿好量子戰衣

出發

美國隊長 + 鋼鐵人 + 蟻人 + 浩克先生-> 復1的紐約

火箭浣熊 + 肥神 -> 雷神2的阿斯嘉

涅布拉 + 戰爭機器 + 鷹眼 + 黑寡婦 -> 星際異工隊1

總覺得最後一個分組怪怪的 怎麼會給三個人類去外星球搶東西= =





復1的紐約

浩克看到自己以前的樣子 學著砸了一些東西的場景真的很逗趣XD

然後他去找當時守護寶石的古一 開始了一串辯論

浩克先生這集真的沒甚麼武戲 都在講話

然後隊長走進電梯 場景跟美2一模一樣

本來以為要開扁了 結果居然正直的美隊以一句Hail Hydra輕鬆A走了權杖

順便致敬漫畫 XD

也順勢解釋為什麼復1之後權杖又跑去九頭蛇那 做實驗做出了緋紅女巫&快銀

鋼鐵人&蟻人 本來要搶到空間魔方 結果來一個浩克討厭走樓梯XD

寶石就這樣被洛基A走順便烙跑

然後隊長走著走著遇到了自己

以為未來隊長是洛基的過去隊長 開始對A

然後又來一句隊長經典台詞

I can do this all day , 我知道XD

終究是自己了解自己 以一句巴奇還活著影響自己 順利尻飛過去隊長

然後權杖戳下去 帶走權杖

會合後再度前往1970 去找還在神盾局的魔方&皮姆

這邊偷東西就沒甚麼意外 再有意外就演不完了XD

這邊的主軸 給了鋼鐵人與爸爸的相會

為人父的鋼鐵人有太多事想跟爸爸講 終於圓夢

以及隊長看卡特 為結局埋了伏筆

原來東尼本來不叫東尼XD

阿斯嘉這邊

肥神各種雷 回到過去還在找酒

然後看到媽媽就忍不住想要回去找媽媽

完全不顧大局 單就這個行為真的很雷

只是這集就是給三本柱圓夢的 當然要給肥神回去找媽媽聊天

搶寶石就交給火箭 他也很輕鬆地搶到了寶石

然後圓完雷神的夢

雷神把喵喵槌叫了過來 帶回未來(當下OS是他把槌子拿走了 待會過去的雷神要怎麼用啊

不過後面就知道雷神回到過去最重要的其實就是把槌子拿走 哈哈哈哈





星際異攻隊這邊

鷹眼跟黑寡婦搭著太空船飛到了佛米爾星

話說 哪來的太空船

而且給兩個人類四處太空旅行 每個星球都有氧氣來著= =??

然後又遇到了守門員紅骷顱

接著開始誰要跳 鷹眼你射箭阻止黑寡婦的時候壓根就想炸死他吧XDDD

然後兩個人一起跳下去的時候

我以為會出現老梗之 互相珍惜所愛的人 並且能為對方而死

才是得到靈魂寶石的正確方法

結果是出現另一個老梗 下面的人鬆手往下跳

黑寡婦真的死了!? 靈魂寶石GET (當下想說不是有個人電影 後來才知道是前傳

涅布拉跟戰爭機器

一拳KO星爵就拿到寶石了

只是本篇最雷的涅布拉不小心跟過去涅布拉連結在一起

然後就被抓了 各種劇透給2014薩諾斯

知道復仇者在過去偷寶石之後 決定跟著到未來

因為寶石已經有人幫忙收集好了

只是 皮姆粒子不是只有一個人來回

後來是怎麼讓2014涅布拉去2023 又可以開啟量子通道給薩諾斯大軍過來呢??

畫面回到2023 過去不管多久 未來其實只過了一秒

所以也能解釋涅布拉被換了也不知道 因為未來沒有比較晚回來



接著把寶石裝上手套 (說好的手套要由矮人做才能發揮威力勒~~~

關在小房間 準備彈指

為啥不把捏布拉也關進來 放他在外面開量子通道 嘖嘖

順利彈指 浩克先生差點就掛了

外面鳥開始飛 鷹眼的手機響了

蟻人開始開心的說我們做到了!

只是看電影的大家都知道 糟了

接著總部就被各種轟炸

薩諾斯坐在外面等著倖存的復仇者出來

然後 三本柱 帥氣登場

開始三英戰呂布

只是薩諾斯就算沒手套也是很強

肥神應該是因為變太肥了 感覺很弱

還被壓著差點被自己的風暴毀滅者戳死

這時

隊長

舉起喵喵槌拉!!!!!!!

超帥 不解釋 看得我都泛淚了

然後帥打一波 還有槌子+盾牌合體技

只是帥打一波之後又被屌虐一波回來

連盾牌都被打飛一半

還被嘴+大軍壓境

隊長渾身傷

只默默拉緊盾牌的帶子

再次地站了起來

雖千萬人吾往矣阿!!!!

超帥的 帥到我又泛淚了

這時絕望感來到最高

一句 Cap do you copy! On your left

然後火圈開始冒出

終於

全部人都回來了

Avengers assemble

接著開始各種秀

鋼鐵小辣椒 女武神天馬 黃蜂女 黑豹 奇異博士 小蜘蛛

手套接力賽其實只是秀大家技能的時候

隊長跟索爾交換武器 索爾還說你拿小的就好

葛魔拉怒踢星爵蛋蛋 涅布拉說你只有他跟樹可以選

然後緋紅女巫屌打2014薩諾斯 只是2014薩諾斯表示你誰啊XD

2014薩諾斯表示: 我啥事都沒做 干我屁事XD

被女巫打得受不了了 惱羞開戰艦砲轟地面

還在想這台煩人的戰艦怎麼辦 就有道紅光來解決一切

驚奇隊長終於飛回來了- -

都快打完了還不回來

打飛戰艦後

給驚奇隊長跟薩諾斯單挑一波

寶石原來還可以單拿起來轟人 只能說薩諾斯真的很會用寶石

接著就是東尼看了正在治水的奇異博士

博士只默默伸出食指

然後東尼就衝去搶手套了

本來以為手套又搶失敗了

薩諾斯拉好手套 這次沒人阻止他了

薩諾斯: I'm Inevitable

!? 啥事都沒發生

只見東尼舉起手 寶石已經全都A過來了











                           東尼 : I am Iron Man











一彈指 一切就結束了

只是鋼鐵人 就這樣華麗的謝幕了

其實繼續打下去 只要手套不被薩諾斯拿走

感覺復仇者方應該還是能贏

應該只是要給鋼鐵人一個豪華的便當才這樣的吧?

後面就是各英雄回歸自己圓滿的結局 (東尼QQ

然後隊長要歸還無限寶石 以免其他時空混亂

54321之後 想不到居然沒回來

看起來只有巴奇不緊張 然後回頭一個老人

是回到過去過平靜生活的隊長 然後將盾牌與美國隊長 交接給了獵鷹

隊長的那支舞 終於跳到了

三本柱之所以如此令人感動

終究是漫威十年來各個電影細膩的刻畫這三個腳色

美國隊長 - 美國隊長123  復仇者1234

鋼鐵人 - 鋼鐵人123 復仇者1234

雷神 - 雷神123 復仇者1234

三個人都用了七部電影 細細刻畫著這三個腳色

美隊從過去 到現在 從對抗納粹 到現在對抗不同的外星人 堅持著自己的正義

鋼鐵人從一個商人 變成了保衛地球 宇宙的超級英雄 堅持著守護自己與所愛的人

雷神從阿斯嘉的王子 接著失去一切 又重新站起 到最後重新出發

感謝漫威十年如此精采的電影



最後 終究有幾個小小問題 不知道是Bug還是我沒注意 不知道有沒有人看到的

1. 皮姆粒子與量子衣

   2019涅布拉其實只有一組來回的皮姆粒子

   2014涅布拉把皮姆粒子給薩諾斯後 應該就沒辦法回到2019?

   就像美隊跟鐵人 要飛去1970的意思一樣 因為那裏不只有空間魔方 還有皮姆粒子

   而且薩諾斯大軍也都沒有量子衣 這樣到底怎麼飛到未來的= =? (太空船比較高級XD?

   而且回到過去的飛去佛米爾星的太空船又是哪來的?

2. 無限手套

   鋼鐵人很輕鬆地就做出了無限手套 然後把寶石放上去

   浩克博士還嚇了他一下

   復3不是說只有矮人做的無限手套才能發揮寶石的能力嗎?

3. A寶石

   最後東尼A走寶石 應該是因為他手套上有動手腳來著?

   也太好拔了 應該也只是為了華麗便當作準備

4. 美國隊長的圓盾

   不是在大戰時被打壞了一半 怎麼傳給獵鷹的時候又好了?

   雖然我覺得美隊只是純粹圓夢 應該沒有要解釋這段的合理性了

突然想到回來補充

覺得4的薩諾斯相較比較扁平跟壞人

不像3的有自己的理想跟抱負

應該是因為要塑造一個打爆他也不可惜的魔王吧


--
        ''')
        result = phraseg.extract_sent("覺得4的薩諾斯相較比較扁平跟壞人")
        print(result)
        self.assertTrue(len(phraseg.ngrams) > 1)
        self.assertTrue(len(result) > 1)

    def testJapanese(self):
        phraseg = Phraseg('''
        作品の概要
        本作は、22世紀の未来からやってきたネコ型ロボット・ドラえもんと、勉強もスポーツも苦手な小学生・野比のび太が繰り広げる少し不思議（SF）な日常生活を描いた作品である。基本的には一話完結型の連載漫画であるが、一方でストーリー漫画形式となって日常生活を離れた冒険をするという映画版の原作でもある「大長編」シリーズもある。一話完結の基本的なプロットは、「ドラえもんがポケットから出す多種多様なひみつ道具（現代の技術では実現不可能な機能を持つ）で、のび太（以外の場合もある）の身にふりかかった災難を一時的に解決するが、道具を不適切に使い続けた結果、しっぺ返しを受ける」というものが多く、前作の「ウメ星デンカ」のストーリー構図をほぼそのまま踏襲しており実質的な後継作品ともいえる。このプロットは、作者の藤子・F・不二雄が自身のSF作品で描いた独自定義「すこし・不思議」（Sukoshi Fushigi）[注 2]という作風に由来し、当時の一般SF作品の唱える「if」（もしも） についての対象を想定した回答が反映されている。
        作品の主人公はドラえもんであるが、上記のプロットのように物語の主な視点人物はのび太である。
        ''')
        result = phraseg.extract()
        print(result)
        self.assertTrue(len(phraseg.ngrams) > 1)
        self.assertTrue(len(result) > 1)

    def testCantonese(self):
        phraseg = Phraseg('''
        「硬膠」原本是指硬身的塑膠，所以亦有「軟膠」，但在網上有人用此詞取其諧音代替戇鳩這個粵語粗口；直譯英語為 Hard Plastic，繼而引申出縮寫 HP。此用詞是諧音粗口，帶有不雅或恥笑成份。部分人不接受說「膠」字，認為膠是粗口之一[1]。
        硬膠亦簡稱「膠」，可作名詞、形容詞、動詞等使用。硬膠雖與「on9」的語調上不同，但意思差不多，有時可以相通。小丑icon和小丑神也是硬膠的形象化的圖像顯示和象徵。
        「硬膠」一詞聽聞歷史悠久，但出處不明，而由香港高登討論區將其發揚光大，現己推展至香港其他主要網絡社區。在2002年時，有些網民利用「戇鳩」的諧音，發明「硬膠」一詞，更把愛發表無厘頭帖子的會員腦魔二世定為「硬膠」始祖。自此，「硬膠文化」便慢慢發展起來，某程度上硬膠文化與無厘頭文化相似。因腦魔二世的「硬膠」功力驚人，更成為當時被剝削的會員的救星，可令他們苦中作樂一番，故有人曾經預言：「救高登，靠膠人」。及後，高登會員以縮寫「膠」來代替硬膠。而高登亦有了「膠登」的綽號。當時甚至出現了7位膠力驚人的高登會員，以腦魔二世為首，合稱為「硬膠七子」。
        其實「硬膠」早於周星馳電影《整蠱專家》早已出現雛型。戲中有一台詞為「超級戇膠膠」，可見膠字是用作取代粵語粗口鳩字。
        有網友提供資料指西方早於上世紀六十年代亦已經將「膠」（Plastics）等同愚蠢：
        ''')
        result = phraseg.extract()
        print(result)
        self.assertTrue(len(phraseg.ngrams) > 1)
        self.assertTrue(len(result) > 1)

    def testKorean(self):
        phraseg = Phraseg('''
        "서울"의 어원에 대해서는 여러 가지 설이 존재하나, 학계에서는 일반적으로 수도(首都)를 뜻하는 신라 계통의 고유어인 서라벌에서 유래했다는 설이 유력하게 받아들이고 있다. 이때 한자 가차 표기인 서라벌이 원래 어떤 의미였을지에 대해서도 여러 학설이 존재한다. 삼국사기 등에서 서라벌을 금성(金城)으로도 표기했다는 점과 신라(新羅)까지 포함하여 "설[새: 新, 金]-벌[땅: 羅, 城]", 즉 '새로운 땅'이라는 뜻으로 새기는 견해가 있다. 고대-중세 한국어에서 서라벌을 정확히 어떻게 발음했는지에 대해서는 확실하게 확인된 바가 없으며, 그 발음을 알 수 있게 되는 것은 훈민정음의 창제 후 "셔ᄫᅳᆯ"이라는 표기가 등장하고 나서부터이다.
        조선 시대에는 서울을 한양 이외에도 경도(京都), 경부(京府), 경사(京師), 경성(京城), 경조(京兆) 등으로 쓰는 경우가 종종 있었으며, 김정호의 수선전도에서 알 수 있듯 수선(首善)으로 표기한 예도 있다. 그 밖의 표기 중에는 서울의 한자 음차 표기로서 박제가가 북학의에서 썼던 '徐蔚(서울)'이 있다. 이는 모두 수도를 뜻하는 일반명사들로서 '서울'이 원래는 서울 지역(사대문 안과 강북의 성저십리)을 가리키는 말이 아닌 수도를 뜻하는 일반명사였다는 방증이다. 국어사전에서는 일반명사 '서울'을 '한 나라의 중앙 정부가 있고, 경제, 문화, 정치 등에서 가장 중심이 되는 도시'라고 정의하고 있다.[4] 1910년 10월 1일에 일제가 한성부를 경성부(京城府)로 개칭하면서 일제강점기에 서울은 주로 경성(京城, 일본어로는 けいじょう)으로 불렸으며, 1945년 광복 후에는 '경성'이란 말은 도태되고 거의 '서울'로 부르게 되었다.[
        ''')
        result = phraseg.extract()
        print(result)
        self.assertTrue(len(phraseg.ngrams) > 1)
        self.assertTrue(len(result) > 1)

    def testFrance(self):
        phraseg = Phraseg('''
        El idioma francés (le français /lə fʁɑ̃sɛ/ ( escuchar) o la langue française /la lɑ̃ɡ fʁɑ̃sɛz/) es una lengua romance hablada en la Unión Europea, especialmente en Francia, país en el que se habla junto con otras lenguas regionales como el idioma bretón (Bretaña), el occitano (Occitania), el vasco (país vasco francés), el catalán (Rosellón), y el corso (Córcega). En los territorios franceses de ultramar es hablado en muchos casos junto con otras lenguas como el tahitiano (Polinesia Francesa), o el créole (isla Reunión, Guadalupe y Martinica). También se habla en Canadá, Estados Unidos (francés cajún, créole y francés acadio o acadiano), Haití (con el créole), y numerosos países del mundo. Según estimaciones de la Organización Internacional de la Francofonía (basadas en proyecciones demográficas de las Naciones Unidas), en el transcurso del s. XXI, el francés se convertiría en el tercer idioma con el mayor número de hablantes del mundo, sobre todo por el crecimiento poblacional de los países africanos francófonos.5​
        ''')
        result = phraseg.extract()
        print(result)
        self.assertTrue(len(phraseg.ngrams) > 1)
        self.assertTrue(len(result) > 1)

    def testSpanish(self):
        phraseg = Phraseg('''
        Breaking Bad es una serie de televisión dramática estadounidense creada y producida por Vince Gilligan. Breaking Bad narra la historia de Walter White (Bryan Cranston), un profesor de química con problemas económicos a quien le diagnostican un cáncer de pulmón inoperable. Para pagar su tratamiento y asegurar el futuro económico de su familia comienza a cocinar y vender metanfetamina,1​ junto con Jesse Pinkman (Aaron Paul), un antiguo alumno suyo. La serie, ambientada y producida en Albuquerque (Nuevo México), se caracteriza por poner a sus personajes en situaciones que aparentemente no tienen salida, lo que llevó a que su creador la describa como un wéstern contemporáneo.2​
        La serie se estrenó el 20 de enero de 2008 y es una producción de Sony Pictures Television. En Estados Unidos y Canadá se emitió por la cadena AMC.3​ La temporada final se dividió en dos partes de ocho episodios cada una y se emitió en el transcurso de dos años: la primera mitad se estrenó el 15 de julio de 2012 y concluyó el 2 de septiembre de 2012, mientras que la segunda mitad se estrenó el 11 de agosto de 2013 y concluyó el 29 de septiembre del mismo año.
        ''')
        result = phraseg.extract()
        print(result)
        self.assertTrue(len(phraseg.ngrams) > 1)
        self.assertTrue(len(result) > 1)

    def testThai(self):
        phraseg = Phraseg('''
        กิตฮับ (อังกฤษ: GitHub) เป็นเว็บบริการพื้นที่ทางอินเทอร์เน็ต (hosting service) สำหรับเก็บการควบคุมการปรับปรุงแก้ไข (version control) โดยใช้กิต (Git) โดยมากจะใช้จัดเก็บรหัสต้นฉบับ (source code) แต่ยังคงคุณสมบัติเดิมของกิตไว้ อย่างการให้สิทธิ์ควบคุมและปรับปรุงแก้ไข (distributed version control) และระบบการจัดการรหัสต้นฉบับรวมถึงทางกิตฮับได้เพิ่มเติมคุณสมบัติอื่นๆผนวกไว้ด้วย เช่น การควบคุมการเข้าถึงรหัสต้นฉบับ (access control) และ คุณสมบัติด้านความร่วมมือเช่น ติดตามข้อบกพร่อง (bug tracking), การร้องขอให้เพิ่มคุณสมบัติอื่นๆ (feature requests), ระบบจัดการงาน (task management) และวิกิสำหรับทุกโครงการ[2]
        ''')
        result = phraseg.extract()
        print(result)
        self.assertTrue(len(phraseg.ngrams) > 1)
        self.assertTrue(len(result) > 1)

    def testLang(self):
        phraseg = Phraseg('''
        ''')
        result = phraseg.extract()
        print(result)
        self.assertTrue(len(phraseg.ngrams) == 0)
        self.assertTrue(len(result)  == 0)

if __name__ == '__main__':
    unittest.Test()
