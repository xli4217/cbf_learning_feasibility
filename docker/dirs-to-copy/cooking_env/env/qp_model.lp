\ Model CBF_CLF_QP
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
 [ 2 x1_input_acceleration ^2 + 2 x2_input_acceleration ^2
   + 2 x3_input_acceleration ^2 + 2 relaxation_CLF ^2 ] / 2 
Subject To
 CBF_Constraint_sphere_left: - 0.1120003461837769 x1_input_acceleration
   - 0.5499987602233887 x2_input_acceleration
   - 0.1100008487701416 x3_input_acceleration >= -0.0511607251306755
 CBF_Constraint_sphere_low: - 0.2120001316070557 x1_input_acceleration
   - 0.9499983787536621 x2_input_acceleration
   + 0.0899989604949951 x3_input_acceleration >= -0.2082601970815438
 CBF_Constraint_sphere_right: - 0.1120001673698425 x1_input_acceleration
   - 1.39999794960022 x2_input_acceleration
   - 0.1100008487701416 x3_input_acceleration >= -0.4655346207764609
 Relaxed_CLF_constraint: - 0.169999748468399 x1_input_acceleration
   - 0.4109988212585449 x2_input_acceleration
   - 0.1940005421638489 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: - 0.1080003380775452 x1_input_acceleration
   - 0.5478925704956055 x2_input_acceleration
   - 0.1060009002685547 x3_input_acceleration >= -0.0501466331717225
 CBF_Constraint_sphere_low: - 0.208000123500824 x1_input_acceleration
   - 0.9478921890258789 x2_input_acceleration
   + 0.093998908996582 x3_input_acceleration >= -0.2070248620712946
 CBF_Constraint_sphere_right: - 0.1080001592636108 x1_input_acceleration
   - 1.397891759872437 x2_input_acceleration
   - 0.1060009002685547 x3_input_acceleration >= -0.4636253993944918
 Relaxed_CLF_constraint: - 0.1679997444152832 x1_input_acceleration
   - 0.4099457263946533 x2_input_acceleration
   - 0.1920005679130554 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: - 0.1040003299713135 x1_input_acceleration
   - 0.5457940101623535 x2_input_acceleration
   - 0.1020009517669678 x3_input_acceleration >= -0.0491528410811532
 CBF_Constraint_sphere_low: - 0.2040001153945923 x1_input_acceleration
   - 0.945793628692627 x2_input_acceleration
   + 0.0979988574981689 x3_input_acceleration >= -0.20581135280688
 CBF_Constraint_sphere_right: - 0.1040001511573792 x1_input_acceleration
   - 1.395793199539185 x2_input_acceleration
   - 0.1020009517669678 x3_input_acceleration >= -0.4617397203704897
 Relaxed_CLF_constraint: - 0.1659997403621674 x1_input_acceleration
   - 0.4088964462280273 x2_input_acceleration
   - 0.190000593662262 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: - 0.1000003218650818 x1_input_acceleration
   - 0.5437016487121582 x2_input_acceleration
   - 0.0980010032653809 x3_input_acceleration >= -0.0481789359566151
 CBF_Constraint_sphere_low: - 0.2000001072883606 x1_input_acceleration
   - 0.9437012672424316 x2_input_acceleration
   + 0.1019988059997559 x3_input_acceleration >= -0.2046189702839258
 CBF_Constraint_sphere_right: - 0.1000001430511475 x1_input_acceleration
   - 1.393700838088989 x2_input_acceleration
   - 0.0980010032653809 x3_input_acceleration >= -0.4598765628353055
 Relaxed_CLF_constraint: - 0.1639997363090515 x1_input_acceleration
   - 0.4078502655029297 x2_input_acceleration
   - 0.1880006194114685 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: - 0.0960003137588501 x1_input_acceleration
   - 0.5416140556335449 x2_input_acceleration
   - 0.0940010547637939 x3_input_acceleration >= -0.04722451094958
 CBF_Constraint_sphere_low: - 0.1960000991821289 x1_input_acceleration
   - 0.9416136741638184 x2_input_acceleration
   + 0.1059987545013428 x3_input_acceleration >= -0.2034470215518815
 CBF_Constraint_sphere_right: - 0.0960001349449158 x1_input_acceleration
   - 1.391613245010376 x2_input_acceleration
   - 0.0940010547637939 x3_input_acceleration >= -0.4580349119736141
 Relaxed_CLF_constraint: - 0.1619997322559357 x1_input_acceleration
   - 0.406806468963623 x2_input_acceleration
   - 0.186000645160675 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: - 0.0920003056526184 x1_input_acceleration
   - 0.5395298004150391 x2_input_acceleration
   - 0.090001106262207 x3_input_acceleration >= -0.046289165226122
 CBF_Constraint_sphere_low: - 0.1920000910758972 x1_input_acceleration
   - 0.9395294189453125 x2_input_acceleration
   + 0.1099987030029297 x3_input_acceleration >= -0.202294819674799
 CBF_Constraint_sphere_right: - 0.0920001268386841 x1_input_acceleration
   - 1.38952898979187 x2_input_acceleration
   - 0.090001106262207 x3_input_acceleration >= -0.4562137589846925
 Relaxed_CLF_constraint: - 0.1599997282028198 x1_input_acceleration
   - 0.4057643413543701 x2_input_acceleration
   - 0.1840006709098816 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: - 0.0880002975463867 x1_input_acceleration
   - 0.537447452545166 x2_input_acceleration
   - 0.0860011577606201 x3_input_acceleration >= -0.045372503937927
 CBF_Constraint_sphere_low: - 0.1880000829696655 x1_input_acceleration
   - 0.9374470710754395 x2_input_acceleration
   + 0.1139986515045166 x3_input_acceleration >= -0.2011616837023424
 CBF_Constraint_sphere_right: - 0.0880001187324524 x1_input_acceleration
   - 1.387446641921997 x2_input_acceleration
   - 0.0860011577606201 x3_input_acceleration >= -0.4544121010534298
 Relaxed_CLF_constraint: - 0.157999724149704 x1_input_acceleration
   - 0.4047231674194336 x2_input_acceleration
   - 0.1820006966590881 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: - 0.084000289440155 x1_input_acceleration
   - 0.5353655815124512 x2_input_acceleration
   - 0.0820012092590332 x3_input_acceleration >= -0.0444741382035346
 CBF_Constraint_sphere_low: - 0.1840000748634338 x1_input_acceleration
   - 0.9353652000427246 x2_input_acceleration
   + 0.1179986000061035 x3_input_acceleration >= -0.200046938651029
 CBF_Constraint_sphere_right: - 0.0840001106262207 x1_input_acceleration
   - 1.385364770889282 x2_input_acceleration
   - 0.0820012092590332 x3_input_acceleration >= -0.4526289413315686
 Relaxed_CLF_constraint: - 0.1559997200965881 x1_input_acceleration
   - 0.4036822319030762 x2_input_acceleration
   - 0.1800007224082947 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: - 0.0800002813339233 x1_input_acceleration
   - 0.5332837104797363 x2_input_acceleration
   - 0.0780012607574463 x3_input_acceleration >= -0.0435939393890733
 CBF_Constraint_sphere_low: - 0.1800000667572021 x1_input_acceleration
   - 0.9332833290100098 x2_input_acceleration
   + 0.1219985485076904 x3_input_acceleration >= -0.1989503605196467
 CBF_Constraint_sphere_right: - 0.080000102519989 x1_input_acceleration
   - 1.383282899856567 x2_input_acceleration
   - 0.0780012607574463 x3_input_acceleration >= -0.4508639485296385
 Relaxed_CLF_constraint: - 0.1539997160434723 x1_input_acceleration
   - 0.4026412963867188 x2_input_acceleration
   - 0.1780007481575012 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: - 0.0760002732276917 x1_input_acceleration
   - 0.5312018394470215 x2_input_acceleration
   - 0.0740013122558594 x3_input_acceleration >= -0.042731907494543
 CBF_Constraint_sphere_low: - 0.1760000586509705 x1_input_acceleration
   - 0.9312014579772949 x2_input_acceleration
   + 0.1259984970092773 x3_input_acceleration >= -0.1978719493081954
 CBF_Constraint_sphere_right: - 0.0760000944137573 x1_input_acceleration
   - 1.381201028823853 x2_input_acceleration
   - 0.0740013122558594 x3_input_acceleration >= -0.4491171226476395
 Relaxed_CLF_constraint: - 0.1519997119903564 x1_input_acceleration
   - 0.4016003608703613 x2_input_acceleration
   - 0.1760007739067078 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: - 0.07200026512146 x1_input_acceleration
   - 0.5291199684143066 x2_input_acceleration
   - 0.0700013637542725 x3_input_acceleration >= -0.0418880425199438
 CBF_Constraint_sphere_low: - 0.1720000505447388 x1_input_acceleration
   - 0.9291195869445801 x2_input_acceleration
   + 0.1299984455108643 x3_input_acceleration >= -0.1968117050166752
 CBF_Constraint_sphere_right: - 0.0720000863075256 x1_input_acceleration
   - 1.379119157791138 x2_input_acceleration
   - 0.0700013637542725 x3_input_acceleration >= -0.4473884636855715
 Relaxed_CLF_constraint: - 0.1499997079372406 x1_input_acceleration
   - 0.4005594253540039 x2_input_acceleration
   - 0.1740007996559143 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: - 0.0680002570152283 x1_input_acceleration
   - 0.5270380973815918 x2_input_acceleration
   - 0.0660014152526855 x3_input_acceleration >= -0.0410623444652757
 CBF_Constraint_sphere_low: - 0.1680000424385071 x1_input_acceleration
   - 0.9270377159118652 x2_input_acceleration
   + 0.1339983940124512 x3_input_acceleration >= -0.1957696276450861
 CBF_Constraint_sphere_right: - 0.0680000782012939 x1_input_acceleration
   - 1.377037286758423 x2_input_acceleration
   - 0.0660014152526855 x3_input_acceleration >= -0.4456779716434346
 Relaxed_CLF_constraint: - 0.1479997038841248 x1_input_acceleration
   - 0.3995184898376465 x2_input_acceleration
   - 0.1720008254051208 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: - 0.0640002489089966 x1_input_acceleration
   - 0.524956226348877 x2_input_acceleration
   - 0.0620014667510986 x3_input_acceleration >= -0.0402548133305386
 CBF_Constraint_sphere_low: - 0.1640000343322754 x1_input_acceleration
   - 0.9249558448791504 x2_input_acceleration
   + 0.1379983425140381 x3_input_acceleration >= -0.1947457171934281
 CBF_Constraint_sphere_right: - 0.0640000700950623 x1_input_acceleration
   - 1.374955415725708 x2_input_acceleration
   - 0.0620014667510986 x3_input_acceleration >= -0.4439856465212287
 Relaxed_CLF_constraint: - 0.1459996998310089 x1_input_acceleration
   - 0.3984775543212891 x2_input_acceleration
   - 0.1700008511543274 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: - 0.0600002408027649 x1_input_acceleration
   - 0.5228743553161621 x2_input_acceleration
   - 0.0580015182495117 x3_input_acceleration >= -0.0394654491157326
 CBF_Constraint_sphere_low: - 0.1600000262260437 x1_input_acceleration
   - 0.9228739738464355 x2_input_acceleration
   + 0.141998291015625 x3_input_acceleration >= -0.1937399736617011
 CBF_Constraint_sphere_right: - 0.0600000619888306 x1_input_acceleration
   - 1.372873544692993 x2_input_acceleration
   - 0.0580015182495117 x3_input_acceleration >= -0.4423114883189539
 Relaxed_CLF_constraint: - 0.1439996957778931 x1_input_acceleration
   - 0.3974366188049316 x2_input_acceleration
   - 0.1680008769035339 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: - 0.0560002326965332 x1_input_acceleration
   - 0.5207924842834473 x2_input_acceleration
   - 0.0540015697479248 x3_input_acceleration >= -0.0386942518208576
 CBF_Constraint_sphere_low: - 0.156000018119812 x1_input_acceleration
   - 0.9207921028137207 x2_input_acceleration
   + 0.1459982395172119 x3_input_acceleration >= -0.1927523970499051
 CBF_Constraint_sphere_right: - 0.0560000538825989 x1_input_acceleration
   - 1.370791673660278 x2_input_acceleration
   - 0.0540015697479248 x3_input_acceleration >= -0.4406554970366102
 Relaxed_CLF_constraint: - 0.1419996917247772 x1_input_acceleration
   - 0.3963956832885742 x2_input_acceleration
   - 0.1660009026527405 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: - 0.0520002245903015 x1_input_acceleration
   - 0.5187106132507324 x2_input_acceleration
   - 0.0500016212463379 x3_input_acceleration >= -0.0379412214459137
 CBF_Constraint_sphere_low: - 0.1520000100135803 x1_input_acceleration
   - 0.9187102317810059 x2_input_acceleration
   + 0.1499981880187988 x3_input_acceleration >= -0.1917829873580403
 CBF_Constraint_sphere_right: - 0.0520000457763672 x1_input_acceleration
   - 1.368709802627563 x2_input_acceleration
   - 0.0500016212463379 x3_input_acceleration >= -0.4390176726741976
 Relaxed_CLF_constraint: - 0.1399996876716614 x1_input_acceleration
   - 0.3953547477722168 x2_input_acceleration
   - 0.164000928401947 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: - 0.0480002164840698 x1_input_acceleration
   - 0.5166287422180176 x2_input_acceleration
   - 0.046001672744751 x3_input_acceleration >= -0.0372063579909009
 CBF_Constraint_sphere_low: - 0.1480000019073486 x1_input_acceleration
   - 0.916628360748291 x2_input_acceleration
   + 0.1539981365203857 x3_input_acceleration >= -0.1908317445861064
 CBF_Constraint_sphere_right: - 0.0480000376701355 x1_input_acceleration
   - 1.366627931594849 x2_input_acceleration
   - 0.046001672744751 x3_input_acceleration >= -0.437398015231716
 Relaxed_CLF_constraint: - 0.1379996836185455 x1_input_acceleration
   - 0.3943138122558594 x2_input_acceleration
   - 0.1620009541511536 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: - 0.0440002083778381 x1_input_acceleration
   - 0.5145468711853027 x2_input_acceleration
   - 0.0420017242431641 x3_input_acceleration >= -0.0364896614558191
 CBF_Constraint_sphere_low: - 0.1439999938011169 x1_input_acceleration
   - 0.9145464897155762 x2_input_acceleration
   + 0.1579980850219727 x3_input_acceleration >= -0.1898986687341037
 CBF_Constraint_sphere_right: - 0.0440000295639038 x1_input_acceleration
   - 1.364546060562134 x2_input_acceleration
   - 0.0420017242431641 x3_input_acceleration >= -0.4357965247091654
 Relaxed_CLF_constraint: - 0.1359996795654297 x1_input_acceleration
   - 0.393272876739502 x2_input_acceleration
   - 0.1600009799003601 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: - 0.0400002002716064 x1_input_acceleration
   - 0.5124650001525879 x2_input_acceleration
   - 0.0380017757415771 x3_input_acceleration >= -0.0357911318406684
 CBF_Constraint_sphere_low: - 0.1399999856948853 x1_input_acceleration
   - 0.9124646186828613 x2_input_acceleration
   + 0.1619980335235596 x3_input_acceleration >= -0.188983759802032
 CBF_Constraint_sphere_right: - 0.0400000214576721 x1_input_acceleration
   - 1.362464189529419 x2_input_acceleration
   - 0.0380017757415771 x3_input_acceleration >= -0.4342132011065459
 Relaxed_CLF_constraint: - 0.1339996755123138 x1_input_acceleration
   - 0.3922319412231445 x2_input_acceleration
   - 0.1580010056495667 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: - 0.0360001921653748 x1_input_acceleration
   - 0.510383129119873 x2_input_acceleration
   - 0.0340018272399902 x3_input_acceleration >= -0.0351107691454488
 CBF_Constraint_sphere_low: - 0.1359999775886536 x1_input_acceleration
   - 0.9103827476501465 x2_input_acceleration
   + 0.1659979820251465 x3_input_acceleration >= -0.1880870177898914
 CBF_Constraint_sphere_right: - 0.0360000133514404 x1_input_acceleration
   - 1.360382318496704 x2_input_acceleration
   - 0.0340018272399902 x3_input_acceleration >= -0.4326480444238575
 Relaxed_CLF_constraint: - 0.131999671459198 x1_input_acceleration
   - 0.3911910057067871 x2_input_acceleration
   - 0.1560010313987732 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: - 0.0320001840591431 x1_input_acceleration
   - 0.5083012580871582 x2_input_acceleration
   - 0.0300018787384033 x3_input_acceleration >= -0.0344485733701602
 CBF_Constraint_sphere_low: - 0.1319999694824219 x1_input_acceleration
   - 0.9083008766174316 x2_input_acceleration
   + 0.1699979305267334 x3_input_acceleration >= -0.1872084426976818
 CBF_Constraint_sphere_right: - 0.0320000052452087 x1_input_acceleration
   - 1.358300447463989 x2_input_acceleration
   - 0.0300018787384033 x3_input_acceleration >= -0.4311010546611002
 Relaxed_CLF_constraint: - 0.1299996674060822 x1_input_acceleration
   - 0.3901500701904297 x2_input_acceleration
   - 0.1540010571479797 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: - 0.0280001759529114 x1_input_acceleration
   - 0.5062193870544434 x2_input_acceleration
   - 0.0260019302368164 x3_input_acceleration >= -0.0338045445148027
 CBF_Constraint_sphere_low: - 0.1279999613761902 x1_input_acceleration
   - 0.9062190055847168 x2_input_acceleration
   + 0.1739978790283203 x3_input_acceleration >= -0.1863480345254033
 CBF_Constraint_sphere_right: - 0.0279999971389771 x1_input_acceleration
   - 1.356218576431274 x2_input_acceleration
   - 0.0260019302368164 x3_input_acceleration >= -0.4295722318182739
 Relaxed_CLF_constraint: - 0.1279996633529663 x1_input_acceleration
   - 0.3891091346740723 x2_input_acceleration
   - 0.1520010828971863 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: - 0.0240001678466797 x1_input_acceleration
   - 0.5041375160217285 x2_input_acceleration
   - 0.0220019817352295 x3_input_acceleration >= -0.0331786825793762
 CBF_Constraint_sphere_low: - 0.1239999532699585 x1_input_acceleration
   - 0.904137134552002 x2_input_acceleration
   + 0.1779978275299072 x3_input_acceleration >= -0.1855057932730559
 CBF_Constraint_sphere_right: - 0.0239999890327454 x1_input_acceleration
   - 1.35413670539856 x2_input_acceleration
   - 0.0220019817352295 x3_input_acceleration >= -0.4280615758953786
 Relaxed_CLF_constraint: - 0.1259996592998505 x1_input_acceleration
   - 0.3880681991577148 x2_input_acceleration
   - 0.1500011086463928 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: - 0.020000159740448 x1_input_acceleration
   - 0.5020556449890137 x2_input_acceleration
   - 0.0180020332336426 x3_input_acceleration >= -0.0325709875638808
 CBF_Constraint_sphere_low: - 0.1199999451637268 x1_input_acceleration
   - 0.9020552635192871 x2_input_acceleration
   + 0.1819977760314941 x3_input_acceleration >= -0.1846817189406395
 CBF_Constraint_sphere_right: - 0.0199999809265137 x1_input_acceleration
   - 1.352054834365845 x2_input_acceleration
   - 0.0180020332336426 x3_input_acceleration >= -0.4265690868924145
 Relaxed_CLF_constraint: - 0.1239996552467346 x1_input_acceleration
   - 0.3870272636413574 x2_input_acceleration
   - 0.1480011343955994 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: - 0.0160001516342163 x1_input_acceleration
   - 0.4999737739562988 x2_input_acceleration
   - 0.0140020847320557 x3_input_acceleration >= -0.0319814594683164
 CBF_Constraint_sphere_low: - 0.1159999370574951 x1_input_acceleration
   - 0.8999733924865723 x2_input_acceleration
   + 0.1859977245330811 x3_input_acceleration >= -0.1838758115281542
 CBF_Constraint_sphere_right: - 0.015999972820282 x1_input_acceleration
   - 1.34997296333313 x2_input_acceleration
   - 0.0140020847320557 x3_input_acceleration >= -0.4250947648093814
 Relaxed_CLF_constraint: - 0.1219996511936188 x1_input_acceleration
   - 0.385986328125 x2_input_acceleration
   - 0.1460011601448059 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: - 0.0120001435279846 x1_input_acceleration
   - 0.497891902923584 x2_input_acceleration
   - 0.0100021362304688 x3_input_acceleration >= -0.0314100982926832
 CBF_Constraint_sphere_low: - 0.1119999289512634 x1_input_acceleration
   - 0.8978915214538574 x2_input_acceleration
   + 0.189997673034668 x3_input_acceleration >= -0.1830880710355999
 CBF_Constraint_sphere_right: - 0.0119999647140503 x1_input_acceleration
   - 1.347891092300415 x2_input_acceleration
   - 0.0100021362304688 x3_input_acceleration >= -0.4236386096462793
 Relaxed_CLF_constraint: - 0.1199996471405029 x1_input_acceleration
   - 0.3849453926086426 x2_input_acceleration
   - 0.1440011858940125 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: - 0.00800013542175293 x1_input_acceleration
   - 0.4958100318908691 x2_input_acceleration
   - 0.00600218772888184 x3_input_acceleration >= -0.030856904036981
 CBF_Constraint_sphere_low: - 0.1079999208450317 x1_input_acceleration
   - 0.8958096504211426 x2_input_acceleration
   + 0.1939976215362549 x3_input_acceleration >= -0.1823184974629767
 CBF_Constraint_sphere_right: - 0.0079999566078186 x1_input_acceleration
   - 1.3458092212677 x2_input_acceleration
   - 0.00600218772888184 x3_input_acceleration >= -0.4222006214031083
 Relaxed_CLF_constraint: - 0.1179996430873871 x1_input_acceleration
   - 0.3839044570922852 x2_input_acceleration
   - 0.142001211643219 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: - 0.00400012731552124 x1_input_acceleration
   - 0.4937281608581543 x2_input_acceleration
   - 0.00200223922729492 x3_input_acceleration >= -0.0303218767012098
 CBF_Constraint_sphere_low: - 0.1039999127388 x1_input_acceleration
   - 0.8937277793884277 x2_input_acceleration
   + 0.1979975700378418 x3_input_acceleration >= -0.1815670908102846
 CBF_Constraint_sphere_right: - 0.00399994850158691 x1_input_acceleration
   - 1.343727350234985 x2_input_acceleration
   - 0.00200223922729492 x3_input_acceleration >= -0.4207808000798684
 Relaxed_CLF_constraint: - 0.1159996390342712 x1_input_acceleration
   - 0.3828635215759277 x2_input_acceleration
   - 0.1400012373924255 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: - 1.19209e-07 x1_input_acceleration
   - 0.4916462898254395 x2_input_acceleration
   + 0.00199770927429199 x3_input_acceleration >= -0.0298050162853697
 CBF_Constraint_sphere_low: - 0.0999999046325684 x1_input_acceleration
   - 0.8916459083557129 x2_input_acceleration
   + 0.2019975185394287 x3_input_acceleration >= -0.1808338510775235
 CBF_Constraint_sphere_right: 5.96046e-08 x1_input_acceleration
   - 1.341645479202271 x2_input_acceleration
   + 0.00199770927429199 x3_input_acceleration >= -0.4193791456765595
 Relaxed_CLF_constraint: - 0.1139996349811554 x1_input_acceleration
   - 0.3818225860595703 x2_input_acceleration
   - 0.1380012631416321 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.00399988889694214 x1_input_acceleration
   - 0.4895644187927246 x2_input_acceleration
   + 0.00599765777587891 x3_input_acceleration >= -0.0293063227894607
 CBF_Constraint_sphere_low: - 0.0959998965263367 x1_input_acceleration
   - 0.889564037322998 x2_input_acceleration
   + 0.2059974670410156 x3_input_acceleration >= -0.1801187782646935
 CBF_Constraint_sphere_right: 0.00400006771087646 x1_input_acceleration
   - 1.339563608169556 x2_input_acceleration
   + 0.00599765777587891 x3_input_acceleration >= -0.4179956581931817
 Relaxed_CLF_constraint: - 0.1119996309280396 x1_input_acceleration
   - 0.3807816505432129 x2_input_acceleration
   - 0.1360012888908386 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.00799989700317383 x1_input_acceleration
   - 0.4874825477600098 x2_input_acceleration
   + 0.00999760627746582 x3_input_acceleration >= -0.0288257962134827
 CBF_Constraint_sphere_low: - 0.091999888420105 x1_input_acceleration
   - 0.8874821662902832 x2_input_acceleration
   + 0.2099974155426025 x3_input_acceleration >= -0.1794218723717945
 CBF_Constraint_sphere_right: 0.00800007581710815 x1_input_acceleration
   - 1.337481737136841 x2_input_acceleration
   + 0.00999760627746582 x3_input_acceleration >= -0.416630337629735
 Relaxed_CLF_constraint: - 0.1099996268749237 x1_input_acceleration
   - 0.3797407150268555 x2_input_acceleration
   - 0.1340013146400452 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.0119999051094055 x1_input_acceleration
   - 0.4854006767272949 x2_input_acceleration
   + 0.0139975547790527 x3_input_acceleration >= -0.0283634365574358
 CBF_Constraint_sphere_low: - 0.0879998803138733 x1_input_acceleration
   - 0.8854002952575684 x2_input_acceleration
   + 0.2139973640441895 x3_input_acceleration >= -0.1787431333988267
 CBF_Constraint_sphere_right: 0.0120000839233398 x1_input_acceleration
   - 1.335399866104126 x2_input_acceleration
   + 0.0139975547790527 x3_input_acceleration >= -0.4152831839862193
 Relaxed_CLF_constraint: - 0.1079996228218079 x1_input_acceleration
   - 0.378699779510498 x2_input_acceleration
   - 0.1320013403892517 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.0159999132156372 x1_input_acceleration
   - 0.4833188056945801 x2_input_acceleration
   + 0.0179975032806396 x3_input_acceleration >= -0.02791924382132
 CBF_Constraint_sphere_low: - 0.0839998722076416 x1_input_acceleration
   - 0.8833184242248535 x2_input_acceleration
   + 0.2179973125457764 x3_input_acceleration >= -0.1780825613457898
 CBF_Constraint_sphere_right: 0.0160000920295715 x1_input_acceleration
   - 1.333317995071411 x2_input_acceleration
   + 0.0179975032806396 x3_input_acceleration >= -0.4139541972626347
 Relaxed_CLF_constraint: - 0.105999618768692 x1_input_acceleration
   - 0.3776588439941406 x2_input_acceleration
   - 0.1300013661384583 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.0199999213218689 x1_input_acceleration
   - 0.4812369346618652 x2_input_acceleration
   + 0.0219974517822266 x3_input_acceleration >= -0.0274932180051352
 CBF_Constraint_sphere_low: - 0.0799998641014099 x1_input_acceleration
   - 0.8812365531921387 x2_input_acceleration
   + 0.2219972610473633 x3_input_acceleration >= -0.1774401562126841
 CBF_Constraint_sphere_right: 0.0200001001358032 x1_input_acceleration
   - 1.331236124038696 x2_input_acceleration
   + 0.0219974517822266 x3_input_acceleration >= -0.4126433774589812
 Relaxed_CLF_constraint: - 0.1039996147155762 x1_input_acceleration
   - 0.3766179084777832 x2_input_acceleration
   - 0.1280013918876648 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.0239999294281006 x1_input_acceleration
   - 0.4791550636291504 x2_input_acceleration
   + 0.0259974002838135 x3_input_acceleration >= -0.0270853591088815
 CBF_Constraint_sphere_low: - 0.0759998559951782 x1_input_acceleration
   - 0.8791546821594238 x2_input_acceleration
   + 0.2259972095489502 x3_input_acceleration >= -0.1768159179995094
 CBF_Constraint_sphere_right: 0.0240001082420349 x1_input_acceleration
   - 1.329154253005981 x2_input_acceleration
   + 0.0259974002838135 x3_input_acceleration >= -0.4113507245752587
 Relaxed_CLF_constraint: - 0.1019996106624603 x1_input_acceleration
   - 0.3755769729614258 x2_input_acceleration
   - 0.1260014176368713 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.0279999375343323 x1_input_acceleration
   - 0.4770731925964355 x2_input_acceleration
   + 0.0299973487854004 x3_input_acceleration >= -0.0266956671325588
 CBF_Constraint_sphere_low: - 0.0719998478889465 x1_input_acceleration
   - 0.877072811126709 x2_input_acceleration
   + 0.2299971580505371 x3_input_acceleration >= -0.1762098467062657
 CBF_Constraint_sphere_right: 0.0280001163482666 x1_input_acceleration
   - 1.327072381973267 x2_input_acceleration
   + 0.0299973487854004 x3_input_acceleration >= -0.4100762386114672
 Relaxed_CLF_constraint: - 0.0999996066093445 x1_input_acceleration
   - 0.3745360374450684 x2_input_acceleration
   - 0.1240014433860779 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.031999945640564 x1_input_acceleration
   - 0.4749913215637207 x2_input_acceleration
   + 0.0339972972869873 x3_input_acceleration >= -0.0263241420761672
 CBF_Constraint_sphere_low: - 0.0679998397827148 x1_input_acceleration
   - 0.8749909400939941 x2_input_acceleration
   + 0.233997106552124 x3_input_acceleration >= -0.1756219423329532
 CBF_Constraint_sphere_right: 0.0320001244544983 x1_input_acceleration
   - 1.324990510940552 x2_input_acceleration
   + 0.0339972972869873 x3_input_acceleration >= -0.4088199195676069
 Relaxed_CLF_constraint: - 0.0979996025562286 x1_input_acceleration
   - 0.3734951019287109 x2_input_acceleration
   - 0.1220014691352844 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.0359999537467957 x1_input_acceleration
   - 0.4729094505310059 x2_input_acceleration
   + 0.0379972457885742 x3_input_acceleration >= -0.0259707839397067
 CBF_Constraint_sphere_low: - 0.0639998316764832 x1_input_acceleration
   - 0.8729090690612793 x2_input_acceleration
   + 0.2379970550537109 x3_input_acceleration >= -0.1750522048795717
 CBF_Constraint_sphere_right: 0.03600013256073 x1_input_acceleration
   - 1.322908639907837 x2_input_acceleration
   + 0.0379972457885742 x3_input_acceleration >= -0.4075817674436776
 Relaxed_CLF_constraint: - 0.0959995985031128 x1_input_acceleration
   - 0.3724541664123535 x2_input_acceleration
   - 0.120001494884491 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.0399999618530273 x1_input_acceleration
   - 0.470827579498291 x2_input_acceleration
   + 0.0419971942901611 x3_input_acceleration >= -0.0256355927231772
 CBF_Constraint_sphere_low: - 0.0599998235702515 x1_input_acceleration
   - 0.8708271980285645 x2_input_acceleration
   + 0.2419970035552979 x3_input_acceleration >= -0.1745006343461212
 CBF_Constraint_sphere_right: 0.0400001406669617 x1_input_acceleration
   - 1.320826768875122 x2_input_acceleration
   + 0.0419971942901611 x3_input_acceleration >= -0.4063617822396793
 Relaxed_CLF_constraint: - 0.0939995944499969 x1_input_acceleration
   - 0.3714132308959961 x2_input_acceleration
   - 0.1180015206336975 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.043999969959259 x1_input_acceleration
   - 0.4687457084655762 x2_input_acceleration
   + 0.045997142791748 x3_input_acceleration >= -0.0253185684265788
 CBF_Constraint_sphere_low: - 0.0559998154640198 x1_input_acceleration
   - 0.8687453269958496 x2_input_acceleration
   + 0.2459969520568848 x3_input_acceleration >= -0.1739672307326018
 CBF_Constraint_sphere_right: 0.0440001487731934 x1_input_acceleration
   - 1.318744897842407 x2_input_acceleration
   + 0.045997142791748 x3_input_acceleration >= -0.4051599639556122
 Relaxed_CLF_constraint: - 0.0919995903968811 x1_input_acceleration
   - 0.3703722953796387 x2_input_acceleration
   - 0.1160015463829041 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.0479999780654907 x1_input_acceleration
   - 0.4666638374328613 x2_input_acceleration
   + 0.049997091293335 x3_input_acceleration >= -0.0250197110499114
 CBF_Constraint_sphere_low: - 0.0519998073577881 x1_input_acceleration
   - 0.8666634559631348 x2_input_acceleration
   + 0.2499969005584717 x3_input_acceleration >= -0.1734519940390135
 CBF_Constraint_sphere_right: 0.048000156879425 x1_input_acceleration
   - 1.316663026809692 x2_input_acceleration
   + 0.049997091293335 x3_input_acceleration >= -0.403976312591476
 Relaxed_CLF_constraint: - 0.0899995863437653 x1_input_acceleration
   - 0.3693313598632812 x2_input_acceleration
   - 0.1140015721321106 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.0519999861717224 x1_input_acceleration
   - 0.4645819664001465 x2_input_acceleration
   + 0.0539970397949219 x3_input_acceleration >= -0.0247390205931751
 CBF_Constraint_sphere_low: - 0.0479997992515564 x1_input_acceleration
   - 0.8645815849304199 x2_input_acceleration
   + 0.2539968490600586 x3_input_acceleration >= -0.1729549242653562
 CBF_Constraint_sphere_right: 0.0520001649856567 x1_input_acceleration
   - 1.314581155776978 x2_input_acceleration
   + 0.0539970397949219 x3_input_acceleration >= -0.402810828147271
 Relaxed_CLF_constraint: - 0.0879995822906494 x1_input_acceleration
   - 0.3682904243469238 x2_input_acceleration
   - 0.1120015978813171 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.0559999942779541 x1_input_acceleration
   - 0.4625000953674316 x2_input_acceleration
   + 0.0579969882965088 x3_input_acceleration >= -0.0244764970563699
 CBF_Constraint_sphere_low: - 0.0439997911453247 x1_input_acceleration
   - 0.8624997138977051 x2_input_acceleration
   + 0.2579967975616455 x3_input_acceleration >= -0.17247602141163
 CBF_Constraint_sphere_right: 0.0560001730918884 x1_input_acceleration
   - 1.312499284744263 x2_input_acceleration
   + 0.0579969882965088 x3_input_acceleration >= -0.401663510622997
 Relaxed_CLF_constraint: - 0.0859995782375336 x1_input_acceleration
   - 0.3672494888305664 x2_input_acceleration
   - 0.1100016236305237 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.0600000023841858 x1_input_acceleration
   - 0.4604182243347168 x2_input_acceleration
   + 0.0619969367980957 x3_input_acceleration >= -0.0242321404394957
 CBF_Constraint_sphere_low: - 0.039999783039093 x1_input_acceleration
   - 0.8604178428649902 x2_input_acceleration
   + 0.2619967460632324 x3_input_acceleration >= -0.1720152854778349
 CBF_Constraint_sphere_right: 0.0600001811981201 x1_input_acceleration
   - 1.310417413711548 x2_input_acceleration
   + 0.0619969367980957 x3_input_acceleration >= -0.4005343600186541
 Relaxed_CLF_constraint: - 0.0839995741844177 x1_input_acceleration
   - 0.366208553314209 x2_input_acceleration
   - 0.1080016493797302 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.0640000104904175 x1_input_acceleration
   - 0.458336353302002 x2_input_acceleration
   + 0.0659968852996826 x3_input_acceleration >= -0.0240059507425526
 CBF_Constraint_sphere_low: - 0.0359997749328613 x1_input_acceleration
   - 0.8583359718322754 x2_input_acceleration
   + 0.2659966945648193 x3_input_acceleration >= -0.1715727164639708
 CBF_Constraint_sphere_right: 0.0640001893043518 x1_input_acceleration
   - 1.308335542678833 x2_input_acceleration
   + 0.0659968852996826 x3_input_acceleration >= -0.3994233763342422
 Relaxed_CLF_constraint: - 0.0819995701313019 x1_input_acceleration
   - 0.3651676177978516 x2_input_acceleration
   - 0.1060016751289368 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.0680000185966492 x1_input_acceleration
   - 0.4562544822692871 x2_input_acceleration
   + 0.0699968338012695 x3_input_acceleration >= -0.0237979279655406
 CBF_Constraint_sphere_low: - 0.0319997668266296 x1_input_acceleration
   - 0.8562541007995605 x2_input_acceleration
   + 0.2699966430664062 x3_input_acceleration >= -0.1711483143700378
 CBF_Constraint_sphere_right: 0.0680001974105835 x1_input_acceleration
   - 1.306253671646118 x2_input_acceleration
   + 0.0699968338012695 x3_input_acceleration >= -0.3983305595697614
 Relaxed_CLF_constraint: - 0.079999566078186 x1_input_acceleration
   - 0.3641266822814941 x2_input_acceleration
   - 0.1040017008781433 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.0720000267028809 x1_input_acceleration
   - 0.4541726112365723 x2_input_acceleration
   + 0.0739967823028564 x3_input_acceleration >= -0.0236080721084596
 CBF_Constraint_sphere_low: - 0.0279997587203979 x1_input_acceleration
   - 0.8541722297668457 x2_input_acceleration
   + 0.2739965915679932 x3_input_acceleration >= -0.1707420791960358
 CBF_Constraint_sphere_right: 0.0720002055168152 x1_input_acceleration
   - 1.304171800613403 x2_input_acceleration
   + 0.0739967823028564 x3_input_acceleration >= -0.3972559097252116
 Relaxed_CLF_constraint: - 0.0779995620250702 x1_input_acceleration
   - 0.3630857467651367 x2_input_acceleration
   - 0.1020017266273499 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.0760000348091125 x1_input_acceleration
   - 0.4520907402038574 x2_input_acceleration
   + 0.0779967308044434 x3_input_acceleration >= -0.0234363831713097
 CBF_Constraint_sphere_low: - 0.0239997506141663 x1_input_acceleration
   - 0.8520903587341309 x2_input_acceleration
   + 0.2779965400695801 x3_input_acceleration >= -0.1703540109419649
 CBF_Constraint_sphere_right: 0.0760002136230469 x1_input_acceleration
   - 1.302089929580688 x2_input_acceleration
   + 0.0779967308044434 x3_input_acceleration >= -0.3961994268005929
 Relaxed_CLF_constraint: - 0.0759995579719543 x1_input_acceleration
   - 0.3620448112487793 x2_input_acceleration
   - 0.1000017523765564 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.0800000429153442 x1_input_acceleration
   - 0.4500088691711426 x2_input_acceleration
   + 0.0819966793060303 x3_input_acceleration >= -0.0232828611540909
 CBF_Constraint_sphere_low: - 0.0199997425079346 x1_input_acceleration
   - 0.850008487701416 x2_input_acceleration
   + 0.281996488571167 x3_input_acceleration >= -0.1699841096078251
 CBF_Constraint_sphere_right: 0.0800002217292786 x1_input_acceleration
   - 1.300008058547974 x2_input_acceleration
   + 0.0819966793060303 x3_input_acceleration >= -0.3951611107959053
 Relaxed_CLF_constraint: - 0.0739995539188385 x1_input_acceleration
   - 0.3610038757324219 x2_input_acceleration
   - 0.0980017781257629 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.0840000510215759 x1_input_acceleration
   - 0.4479269981384277 x2_input_acceleration
   + 0.0859966278076172 x3_input_acceleration >= -0.0231475060568031
 CBF_Constraint_sphere_low: - 0.0159997344017029 x1_input_acceleration
   - 0.8479266166687012 x2_input_acceleration
   + 0.2859964370727539 x3_input_acceleration >= -0.1696323751936163
 CBF_Constraint_sphere_right: 0.0840002298355103 x1_input_acceleration
   - 1.297926187515259 x2_input_acceleration
   + 0.0859966278076172 x3_input_acceleration >= -0.3941409617111488
 Relaxed_CLF_constraint: - 0.0719995498657227 x1_input_acceleration
   - 0.3599629402160645 x2_input_acceleration
   - 0.0960018038749695 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.0880000591278076 x1_input_acceleration
   - 0.4458451271057129 x2_input_acceleration
   + 0.0899965763092041 x3_input_acceleration >= -0.0230303178794463
 CBF_Constraint_sphere_low: - 0.0119997262954712 x1_input_acceleration
   - 0.8458447456359863 x2_input_acceleration
   + 0.2899963855743408 x3_input_acceleration >= -0.1692988076993386
 CBF_Constraint_sphere_right: 0.0880002379417419 x1_input_acceleration
   - 1.295844316482544 x2_input_acceleration
   + 0.0899965763092041 x3_input_acceleration >= -0.3931389795463233
 Relaxed_CLF_constraint: - 0.0699995458126068 x1_input_acceleration
   - 0.358922004699707 x2_input_acceleration
   - 0.094001829624176 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.0920000672340393 x1_input_acceleration
   - 0.443763256072998 x2_input_acceleration
   + 0.093996524810791 x3_input_acceleration >= -0.0229312966220207
 CBF_Constraint_sphere_low: - 0.0079997181892395 x1_input_acceleration
   - 0.8437628746032715 x2_input_acceleration
   + 0.2939963340759277 x3_input_acceleration >= -0.168983407124992
 CBF_Constraint_sphere_right: 0.0920002460479736 x1_input_acceleration
   - 1.293762445449829 x2_input_acceleration
   + 0.093996524810791 x3_input_acceleration >= -0.3921551643014288
 Relaxed_CLF_constraint: - 0.067999541759491 x1_input_acceleration
   - 0.3578810691833496 x2_input_acceleration
   - 0.0920018553733826 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.096000075340271 x1_input_acceleration
   - 0.4416813850402832 x2_input_acceleration
   + 0.0979964733123779 x3_input_acceleration >= -0.0228504422845261
 CBF_Constraint_sphere_low: - 0.00399971 x1_input_acceleration
   - 0.8416810035705566 x2_input_acceleration
   + 0.2979962825775146 x3_input_acceleration >= -0.1686861734705764
 CBF_Constraint_sphere_right: 0.0960002541542053 x1_input_acceleration
   - 1.291680574417114 x2_input_acceleration
   + 0.0979964733123779 x3_input_acceleration >= -0.3911895159764655
 Relaxed_CLF_constraint: - 0.0659995377063751 x1_input_acceleration
   - 0.3568401336669922 x2_input_acceleration
   - 0.0900018811225891 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.1000000834465027 x1_input_acceleration
   - 0.4395995140075684 x2_input_acceleration
   + 0.1019964218139648 x3_input_acceleration >= -0.0227877548669625
 CBF_Constraint_sphere_low: 2.98023e-07 x1_input_acceleration
   - 0.8395991325378418 x2_input_acceleration
   + 0.3019962310791016 x3_input_acceleration >= -0.1684071067360919
 CBF_Constraint_sphere_right: 0.100000262260437 x1_input_acceleration
   - 1.289598703384399 x2_input_acceleration
   + 0.1019964218139648 x3_input_acceleration >= -0.3902420345714331
 Relaxed_CLF_constraint: - 0.0639995336532593 x1_input_acceleration
   - 0.3557991981506348 x2_input_acceleration
   - 0.0880019068717957 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.1040000915527344 x1_input_acceleration
   - 0.4375176429748535 x2_input_acceleration
   + 0.1059963703155518 x3_input_acceleration >= -0.02274323436933
 CBF_Constraint_sphere_low: 0.00400030612945557 x1_input_acceleration
   - 0.837517261505127 x2_input_acceleration
   + 0.3059961795806885 x3_input_acceleration >= -0.1681462069215384
 CBF_Constraint_sphere_right: 0.1040002703666687 x1_input_acceleration
   - 1.287516832351685 x2_input_acceleration
   + 0.1059963703155518 x3_input_acceleration >= -0.3893127200863319
 Relaxed_CLF_constraint: - 0.0619995296001434 x1_input_acceleration
   - 0.3547582626342773 x2_input_acceleration
   - 0.0860019326210022 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.1078909635543823 x1_input_acceleration
   - 0.4354178905487061 x2_input_acceleration
   + 0.1099963188171387 x3_input_acceleration >= -0.0227070973949749
 CBF_Constraint_sphere_low: 0.00789117813110352 x1_input_acceleration
   - 0.8354175090789795 x2_input_acceleration
   + 0.3099961280822754 x3_input_acceleration >= -0.1678955711485065
 CBF_Constraint_sphere_right: 0.1078911423683167 x1_input_acceleration
   - 1.285417079925537 x2_input_acceleration
   + 0.1099963188171387 x3_input_acceleration >= -0.3883841895297891
 Relaxed_CLF_constraint: - 0.0600540935993195 x1_input_acceleration
   - 0.3537083864212036 x2_input_acceleration
   - 0.0840019583702087 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.1117818355560303 x1_input_acceleration
   - 0.4333181381225586 x2_input_acceleration
   + 0.1139962673187256 x3_input_acceleration >= -0.0226887341372196
 CBF_Constraint_sphere_low: 0.0117820501327515 x1_input_acceleration
   - 0.833317756652832 x2_input_acceleration
   + 0.3139960765838623 x3_input_acceleration >= -0.1676627090920745
 CBF_Constraint_sphere_right: 0.1117820143699646 x1_input_acceleration
   - 1.28331732749939 x2_input_acceleration
   + 0.1139962673187256 x3_input_acceleration >= -0.3874734326898462
 Relaxed_CLF_constraint: - 0.0581086575984955 x1_input_acceleration
   - 0.3526585102081299 x2_input_acceleration
   - 0.0820019841194153 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.1154748797416687 x1_input_acceleration
   - 0.4311873912811279 x2_input_acceleration
   + 0.1179962158203125 x3_input_acceleration >= -0.0226700302997728
 CBF_Constraint_sphere_low: 0.0154750943183899 x1_input_acceleration
   - 0.8311870098114014 x2_input_acceleration
   + 0.3179960250854492 x3_input_acceleration >= -0.1674331989483819
 CBF_Constraint_sphere_right: 0.115475058555603 x1_input_acceleration
   - 1.281186580657959 x2_input_acceleration
   + 0.1179962158203125 x3_input_acceleration >= -0.3865491626385916
 Relaxed_CLF_constraint: - 0.0562621355056763 x1_input_acceleration
   - 0.3515931367874146 x2_input_acceleration
   - 0.0800020098686218 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.1189708709716797 x1_input_acceleration
   - 0.4290270805358887 x2_input_acceleration
   + 0.1219961643218994 x3_input_acceleration >= -0.022650342020541
 CBF_Constraint_sphere_low: 0.0189710855484009 x1_input_acceleration
   - 0.8290266990661621 x2_input_acceleration
   + 0.3219959735870361 x3_input_acceleration >= -0.1672066442144212
 CBF_Constraint_sphere_right: 0.118971049785614 x1_input_acceleration
   - 1.27902626991272 x2_input_acceleration
   + 0.1219961643218994 x3_input_acceleration >= -0.3856113434807981
 Relaxed_CLF_constraint: - 0.0545141398906708 x1_input_acceleration
   - 0.3505129814147949 x2_input_acceleration
   - 0.0780020356178284 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.1224668025970459 x1_input_acceleration
   - 0.4268667697906494 x2_input_acceleration
   + 0.1259961128234863 x3_input_acceleration >= -0.022647094334119
 CBF_Constraint_sphere_low: 0.0224670171737671 x1_input_acceleration
   - 0.8268663883209229 x2_input_acceleration
   + 0.325995922088623 x3_input_acceleration >= -0.1669965330534963
 CBF_Constraint_sphere_right: 0.1224669814109802 x1_input_acceleration
   - 1.27686595916748 x2_input_acceleration
   + 0.1259961128234863 x3_input_acceleration >= -0.384689964915809
 Relaxed_CLF_constraint: - 0.0527661740779877 x1_input_acceleration
   - 0.3494328260421753 x2_input_acceleration
   - 0.0760020613670349 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.1257665157318115 x1_input_acceleration
   - 0.4246783256530762 x2_input_acceleration
   + 0.1299960613250732 x3_input_acceleration >= -0.0226419681797131
 CBF_Constraint_sphere_low: 0.0257667303085327 x1_input_acceleration
   - 0.8246779441833496 x2_input_acceleration
   + 0.32999587059021 x3_input_acceleration >= -0.1667887276549648
 CBF_Constraint_sphere_right: 0.1257666945457458 x1_input_acceleration
   - 1.274677515029907 x2_input_acceleration
   + 0.1299960613250732 x3_input_acceleration >= -0.3837547511849536
 Relaxed_CLF_constraint: - 0.0511163175106049 x1_input_acceleration
   - 0.3483386039733887 x2_input_acceleration
   - 0.0740020871162415 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.1290661692619324 x1_input_acceleration
   - 0.4224898815155029 x2_input_acceleration
   + 0.1339960098266602 x3_input_acceleration >= -0.0226526766701
 CBF_Constraint_sphere_low: 0.0290663838386536 x1_input_acceleration
   - 0.8224895000457764 x2_input_acceleration
   + 0.3339958190917969 x3_input_acceleration >= -0.1665967598814519
 CBF_Constraint_sphere_right: 0.1290663480758667 x1_input_acceleration
   - 1.272489070892334 x2_input_acceleration
   + 0.1339960098266602 x3_input_acceleration >= -0.3828353720988856
 Relaxed_CLF_constraint: - 0.0494664907455444 x1_input_acceleration
   - 0.3472443819046021 x2_input_acceleration
   - 0.072002112865448 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.1321703195571899 x1_input_acceleration
   - 0.4202749729156494 x2_input_acceleration
   + 0.1379959583282471 x3_input_acceleration >= -0.0226607326865077
 CBF_Constraint_sphere_low: 0.0321705341339111 x1_input_acceleration
   - 0.8202745914459229 x2_input_acceleration
   + 0.3379957675933838 x3_input_acceleration >= -0.1664066218873194
 CBF_Constraint_sphere_right: 0.1321704983711243 x1_input_acceleration
   - 1.27027416229248 x2_input_acceleration
   + 0.1379959583282471 x3_input_acceleration >= -0.3819020931356164
 Relaxed_CLF_constraint: - 0.0479144155979156 x1_input_acceleration
   - 0.3461369276046753 x2_input_acceleration
   - 0.0700021386146545 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.1351613998413086 x1_input_acceleration
   - 0.4180452823638916 x2_input_acceleration
   + 0.141995906829834 x3_input_acceleration >= -0.0226733249175487
 CBF_Constraint_sphere_low: 0.0351616144180298 x1_input_acceleration
   - 0.818044900894165 x2_input_acceleration
   + 0.3419957160949707 x3_input_acceleration >= -0.1662237172086846
 CBF_Constraint_sphere_right: 0.1351615786552429 x1_input_acceleration
   - 1.268044471740723 x2_input_acceleration
   + 0.141995906829834 x3_input_acceleration >= -0.3809670680533032
 Relaxed_CLF_constraint: - 0.0464188754558563 x1_input_acceleration
   - 0.3450220823287964 x2_input_acceleration
   - 0.0680021643638611 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.138070821762085 x1_input_acceleration
   - 0.4158051013946533 x2_input_acceleration
   + 0.1459958553314209 x3_input_acceleration >= -0.0226930559854571
 CBF_Constraint_sphere_low: 0.0380710363388062 x1_input_acceleration
   - 0.8158047199249268 x2_input_acceleration
   + 0.3459956645965576 x3_input_acceleration >= -0.1660499361948281
 CBF_Constraint_sphere_right: 0.1380710005760193 x1_input_acceleration
   - 1.265804290771484 x2_input_acceleration
   + 0.1459958553314209 x3_input_acceleration >= -0.3800347233773792
 Relaxed_CLF_constraint: - 0.0449641644954681 x1_input_acceleration
   - 0.3439019918441772 x2_input_acceleration
   - 0.0660021901130676 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.1408689618110657 x1_input_acceleration
   - 0.4135513305664062 x2_input_acceleration
   + 0.1499958038330078 x3_input_acceleration >= -0.0227168771456207
 CBF_Constraint_sphere_low: 0.0408691763877869 x1_input_acceleration
   - 0.8135509490966797 x2_input_acceleration
   + 0.3499956130981445 x3_input_acceleration >= -0.1658830913856676
 CBF_Constraint_sphere_right: 0.140869140625 x1_input_acceleration
   - 1.263550519943237 x2_input_acceleration
   + 0.1499958038330078 x3_input_acceleration >= -0.3791006930991904
 Relaxed_CLF_constraint: - 0.0435650944709778 x1_input_acceleration
   - 0.3427751064300537 x2_input_acceleration
   - 0.0640022158622742 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.1435844302177429 x1_input_acceleration
   - 0.4112875461578369 x2_input_acceleration
   + 0.1539957523345947 x3_input_acceleration >= -0.0227471564906466
 CBF_Constraint_sphere_low: 0.0435846447944641 x1_input_acceleration
   - 0.8112871646881104 x2_input_acceleration
   + 0.3539955615997314 x3_input_acceleration >= -0.1657248356204605
 CBF_Constraint_sphere_right: 0.1435846090316772 x1_input_acceleration
   - 1.261286735534668 x2_input_acceleration
   + 0.1539957523345947 x3_input_acceleration >= -0.3781688652308942
 Relaxed_CLF_constraint: - 0.0422073602676392 x1_input_acceleration
   - 0.341643214225769 x2_input_acceleration
   - 0.0620022416114807 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.1462998390197754 x1_input_acceleration
   - 0.4090237617492676 x2_input_acceleration
   + 0.1579957008361816 x3_input_acceleration >= -0.0227916805138625
 CBF_Constraint_sphere_low: 0.0463000535964966 x1_input_acceleration
   - 0.809023380279541 x2_input_acceleration
   + 0.3579955101013184 x3_input_acceleration >= -0.1655808275136691
 CBF_Constraint_sphere_right: 0.1463000178337097 x1_input_acceleration
   - 1.259022951126099 x2_input_acceleration
   + 0.1579957008361816 x3_input_acceleration >= -0.3772512820407826
 Relaxed_CLF_constraint: - 0.0408496558666229 x1_input_acceleration
   - 0.3405113220214844 x2_input_acceleration
   - 0.0600022673606873 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.1488220691680908 x1_input_acceleration
   - 0.4067378044128418 x2_input_acceleration
   + 0.1619956493377686 x3_input_acceleration >= -0.0228315600536041
 CBF_Constraint_sphere_low: 0.048822283744812 x1_input_acceleration
   - 0.8067374229431152 x2_input_acceleration
   + 0.3619954586029053 x3_input_acceleration >= -0.1654373992540214
 CBF_Constraint_sphere_right: 0.1488222479820251 x1_input_acceleration
   - 1.256736993789673 x2_input_acceleration
   + 0.1619956493377686 x3_input_acceleration >= -0.3763196308645732
 Relaxed_CLF_constraint: - 0.0395885407924652 x1_input_acceleration
   - 0.3393683433532715 x2_input_acceleration
   - 0.0580022931098938 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.1513442993164062 x1_input_acceleration
   - 0.404451847076416 x2_input_acceleration
   + 0.1659955978393555 x3_input_acceleration >= -0.0228852330102859
 CBF_Constraint_sphere_low: 0.0513445138931274 x1_input_acceleration
   - 0.8044514656066895 x2_input_acceleration
   + 0.3659954071044922 x3_input_acceleration >= -0.1653077644113138
 CBF_Constraint_sphere_right: 0.1513444781303406 x1_input_acceleration
   - 1.254451036453247 x2_input_acceleration
   + 0.1659955978393555 x3_input_acceleration >= -0.375401773105304
 Relaxed_CLF_constraint: - 0.0383274257183075 x1_input_acceleration
   - 0.3382253646850586 x2_input_acceleration
   - 0.0560023188591003 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.1536741256713867 x1_input_acceleration
   - 0.4021451473236084 x2_input_acceleration
   + 0.1699955463409424 x3_input_acceleration >= -0.0229337355481368
 CBF_Constraint_sphere_low: 0.0536743402481079 x1_input_acceleration
   - 0.8021447658538818 x2_input_acceleration
   + 0.3699953556060791 x3_input_acceleration >= -0.1651784308394794
 CBF_Constraint_sphere_right: 0.153674304485321 x1_input_acceleration
   - 1.252144336700439 x2_input_acceleration
   + 0.1699955463409424 x3_input_acceleration >= -0.3744699293914465
 Relaxed_CLF_constraint: - 0.0371625125408173 x1_input_acceleration
   - 0.3370720148086548 x2_input_acceleration
   - 0.0540023446083069 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.1560039520263672 x1_input_acceleration
   - 0.3998384475708008 x2_input_acceleration
   + 0.1739954948425293 x3_input_acceleration >= -0.0229956123572924
 CBF_Constraint_sphere_low: 0.0560041666030884 x1_input_acceleration
   - 0.7998380661010742 x2_input_acceleration
   + 0.373995304107666 x3_input_acceleration >= -0.1650624715389496
 CBF_Constraint_sphere_right: 0.1560041308403015 x1_input_acceleration
   - 1.249837636947632 x2_input_acceleration
   + 0.1739954948425293 x3_input_acceleration >= -0.3735514599488937
 Relaxed_CLF_constraint: - 0.035997599363327 x1_input_acceleration
   - 0.335918664932251 x2_input_acceleration
   - 0.0520023703575134 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.1581718325614929 x1_input_acceleration
   - 0.3975155353546143 x2_input_acceleration
   + 0.1779954433441162 x3_input_acceleration >= -0.0230548268288488
 CBF_Constraint_sphere_low: 0.0581720471382141 x1_input_acceleration
   - 0.7975151538848877 x2_input_acceleration
   + 0.3779952526092529 x3_input_acceleration >= -0.1649487046848549
 CBF_Constraint_sphere_right: 0.1581720113754272 x1_input_acceleration
   - 1.247514724731445 x2_input_acceleration
   + 0.1779954433441162 x3_input_acceleration >= -0.3726234378638976
 Relaxed_CLF_constraint: - 0.0349136590957642 x1_input_acceleration
   - 0.3347572088241577 x2_input_acceleration
   - 0.05000239610672 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.1603100299835205 x1_input_acceleration
   - 0.3951897621154785 x2_input_acceleration
   + 0.1819953918457031 x3_input_acceleration >= -0.0231241441118192
 CBF_Constraint_sphere_low: 0.0603102445602417 x1_input_acceleration
   - 0.795189380645752 x2_input_acceleration
   + 0.3819952011108398 x3_input_acceleration >= -0.1648459525906004
 CBF_Constraint_sphere_right: 0.1603102087974548 x1_input_acceleration
   - 1.24518895149231 x2_input_acceleration
   + 0.1819953918457031 x3_input_acceleration >= -0.371704302654068
 Relaxed_CLF_constraint: - 0.0338445603847504 x1_input_acceleration
   - 0.3335943222045898 x2_input_acceleration
   - 0.0480024218559265 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.1624481678009033 x1_input_acceleration
   - 0.3928639888763428 x2_input_acceleration
   + 0.18599534034729 x3_input_acceleration >= -0.0232064469021515
 CBF_Constraint_sphere_low: 0.0624483823776245 x1_input_acceleration
   - 0.7928636074066162 x2_input_acceleration
   + 0.3859951496124268 x3_input_acceleration >= -0.1647561889839337
 CBF_Constraint_sphere_right: 0.1624483466148376 x1_input_acceleration
   - 1.242863178253174 x2_input_acceleration
   + 0.18599534034729 x3_input_acceleration >= -0.3707981529515948
 Relaxed_CLF_constraint: - 0.032775491476059 x1_input_acceleration
   - 0.332431435585022 x2_input_acceleration
   - 0.0460024476051331 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.1643953919410706 x1_input_acceleration
   - 0.3905203342437744 x2_input_acceleration
   + 0.189995288848877 x3_input_acceleration >= -0.0232825465335239
 CBF_Constraint_sphere_low: 0.0643956065177917 x1_input_acceleration
   - 0.7905199527740479 x2_input_acceleration
   + 0.3899950981140137 x3_input_acceleration >= -0.1646661916064091
 CBF_Constraint_sphere_right: 0.1643955707550049 x1_input_acceleration
   - 1.240519523620605 x2_input_acceleration
   + 0.189995288848877 x3_input_acceleration >= -0.3698782004881315
 Relaxed_CLF_constraint: - 0.0318018794059753 x1_input_acceleration
   - 0.3312596082687378 x2_input_acceleration
   - 0.0440024733543396 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.1663426160812378 x1_input_acceleration
   - 0.3881766796112061 x2_input_acceleration
   + 0.1939952373504639 x3_input_acceleration >= -0.0233712881583485
 CBF_Constraint_sphere_low: 0.066342830657959 x1_input_acceleration
   - 0.7881762981414795 x2_input_acceleration
   + 0.3939950466156006 x3_input_acceleration >= -0.1645888362223366
 CBF_Constraint_sphere_right: 0.1663427948951721 x1_input_acceleration
   - 1.238175868988037 x2_input_acceleration
   + 0.1939952373504639 x3_input_acceleration >= -0.3689708900181202
 Relaxed_CLF_constraint: - 0.0308282673358917 x1_input_acceleration
   - 0.3300877809524536 x2_input_acceleration
   - 0.0420024991035461 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.168289840221405 x1_input_acceleration
   - 0.3858330249786377 x2_input_acceleration
   + 0.1979951858520508 x3_input_acceleration >= -0.0234726717766251
 CBF_Constraint_sphere_low: 0.0682900547981262 x1_input_acceleration
   - 0.7858326435089111 x2_input_acceleration
   + 0.3979949951171875 x3_input_acceleration >= -0.1645241228317162
 CBF_Constraint_sphere_right: 0.1682900190353394 x1_input_acceleration
   - 1.235832214355469 x2_input_acceleration
   + 0.1979951858520508 x3_input_acceleration >= -0.3680762215415611
 Relaxed_CLF_constraint: - 0.0298546552658081 x1_input_acceleration
   - 0.3289159536361694 x2_input_acceleration
   - 0.0400025248527527 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.1700469255447388 x1_input_acceleration
   - 0.3834731578826904 x2_input_acceleration
   + 0.2019951343536377 x3_input_acceleration >= -0.0235674135015712
 CBF_Constraint_sphere_low: 0.07004714012146 x1_input_acceleration
   - 0.7834727764129639 x2_input_acceleration
   + 0.4019949436187744 x3_input_acceleration >= -0.1644590319786238
 CBF_Constraint_sphere_right: 0.1700471043586731 x1_input_acceleration
   - 1.233472347259521 x2_input_acceleration
   + 0.2019951343536377 x3_input_acceleration >= -0.3671680208643068
 Relaxed_CLF_constraint: - 0.0289761126041412 x1_input_acceleration
   - 0.3277360200881958 x2_input_acceleration
   - 0.0380025506019592 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.1718040108680725 x1_input_acceleration
   - 0.3811132907867432 x2_input_acceleration
   + 0.2059950828552246 x3_input_acceleration >= -0.0236744831812971
 CBF_Constraint_sphere_low: 0.0718042254447937 x1_input_acceleration
   - 0.7811129093170166 x2_input_acceleration
   + 0.4059948921203613 x3_input_acceleration >= -0.1644062690803111
 CBF_Constraint_sphere_right: 0.1718041896820068 x1_input_acceleration
   - 1.231112480163574 x2_input_acceleration
   + 0.2059950828552246 x3_input_acceleration >= -0.3662721481418322
 Relaxed_CLF_constraint: - 0.0280975699424744 x1_input_acceleration
   - 0.3265560865402222 x2_input_acceleration
   - 0.0360025763511658 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.1735610961914062 x1_input_acceleration
   - 0.3787534236907959 x2_input_acceleration
   + 0.2099950313568115 x3_input_acceleration >= -0.0237938808158026
 CBF_Constraint_sphere_low: 0.0735613107681274 x1_input_acceleration
   - 0.7787530422210693 x2_input_acceleration
   + 0.4099948406219482 x3_input_acceleration >= -0.1643658341367781
 CBF_Constraint_sphere_right: 0.1735612750053406 x1_input_acceleration
   - 1.228752613067627 x2_input_acceleration
   + 0.2099950313568115 x3_input_acceleration >= -0.3653886033741373
 Relaxed_CLF_constraint: - 0.0272190272808075 x1_input_acceleration
   - 0.3253761529922485 x2_input_acceleration
   - 0.0340026021003723 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.1752538084983826 x1_input_acceleration
   - 0.3763885498046875 x2_input_acceleration
   + 0.2139949798583984 x3_input_acceleration >= -0.023919022305465
 CBF_Constraint_sphere_low: 0.0752540230751038 x1_input_acceleration
   - 0.7763881683349609 x2_input_acceleration
   + 0.4139947891235352 x3_input_acceleration >= -0.1643333603352361
 CBF_Constraint_sphere_right: 0.1752539873123169 x1_input_acceleration
   - 1.226387739181519 x2_input_acceleration
   + 0.2139949798583984 x3_input_acceleration >= -0.3645086745720546
 Relaxed_CLF_constraint: - 0.0263726711273193 x1_input_acceleration
   - 0.3241937160491943 x2_input_acceleration
   - 0.0320026278495789 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.1768215298652649 x1_input_acceleration
   - 0.3740139007568359 x2_input_acceleration
   + 0.2179949283599854 x3_input_acceleration >= -0.0240435100434781
 CBF_Constraint_sphere_low: 0.0768217444419861 x1_input_acceleration
   - 0.7740135192871094 x2_input_acceleration
   + 0.4179947376251221 x3_input_acceleration >= -0.1643045272851552
 CBF_Constraint_sphere_right: 0.1768217086791992 x1_input_acceleration
   - 1.224013090133667 x2_input_acceleration
   + 0.2179949283599854 x3_input_acceleration >= -0.3636239375673688
 Relaxed_CLF_constraint: - 0.0255888104438782 x1_input_acceleration
   - 0.3230063915252686 x2_input_acceleration
   - 0.0300026535987854 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.1783892512321472 x1_input_acceleration
   - 0.3716392517089844 x2_input_acceleration
   + 0.2219948768615723 x3_input_acceleration >= -0.0241800459296912
 CBF_Constraint_sphere_low: 0.0783894658088684 x1_input_acceleration
   - 0.7716388702392578 x2_input_acceleration
   + 0.421994686126709 x3_input_acceleration >= -0.1642877423832744
 CBF_Constraint_sphere_right: 0.1783894300460815 x1_input_acceleration
   - 1.221638441085815 x2_input_acceleration
   + 0.2219948768615723 x3_input_acceleration >= -0.3627512487108829
 Relaxed_CLF_constraint: - 0.024804949760437 x1_input_acceleration
   - 0.3218190670013428 x2_input_acceleration
   - 0.0280026793479919 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.1799569725990295 x1_input_acceleration
   - 0.3692646026611328 x2_input_acceleration
   + 0.2259948253631592 x3_input_acceleration >= -0.0243286299641042
 CBF_Constraint_sphere_low: 0.0799571871757507 x1_input_acceleration
   - 0.7692642211914062 x2_input_acceleration
   + 0.4259946346282959 x3_input_acceleration >= -0.1642830056295936
 CBF_Constraint_sphere_right: 0.1799571514129639 x1_input_acceleration
   - 1.219263792037964 x2_input_acceleration
   + 0.2259948253631592 x3_input_acceleration >= -0.3618906080025971
 Relaxed_CLF_constraint: - 0.0240210890769958 x1_input_acceleration
   - 0.320631742477417 x2_input_acceleration
   - 0.0260027050971985 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.1813477873802185 x1_input_acceleration
   - 0.3668766021728516 x2_input_acceleration
   + 0.2299834489822388 x3_input_acceleration >= -0.0244694620038409
 CBF_Constraint_sphere_low: 0.0813480019569397 x1_input_acceleration
   - 0.766876220703125 x2_input_acceleration
   + 0.4299832582473755 x3_input_acceleration >= -0.1642755594188309
 CBF_Constraint_sphere_right: 0.1813479661941528 x1_input_acceleration
   - 1.216875791549683 x2_input_acceleration
   + 0.2299834489822388 x3_input_acceleration >= -0.361016540927047
 Relaxed_CLF_constraint: - 0.0233256816864014 x1_input_acceleration
   - 0.3194377422332764 x2_input_acceleration
   - 0.0240083932876587 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.1832380294799805 x1_input_acceleration
   - 0.3644907474517822 x2_input_acceleration
   + 0.2334734201431274 x3_input_acceleration >= -0.0246098795847486
 CBF_Constraint_sphere_low: 0.0832382440567017 x1_input_acceleration
   - 0.7644903659820557 x2_input_acceleration
   + 0.4334732294082642 x3_input_acceleration >= -0.1641932913916625
 CBF_Constraint_sphere_right: 0.1832382082939148 x1_input_acceleration
   - 1.214489936828613 x2_input_acceleration
   + 0.2334734201431274 x3_input_acceleration >= -0.3601429713875156
 Relaxed_CLF_constraint: - 0.0223805606365204 x1_input_acceleration
   - 0.3182448148727417 x2_input_acceleration
   - 0.0222634077072144 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.1853681206703186 x1_input_acceleration
   - 0.3620948791503906 x2_input_acceleration
   + 0.2365366220474243 x3_input_acceleration >= -0.024730903809347
 CBF_Constraint_sphere_low: 0.0853683352470398 x1_input_acceleration
   - 0.7620944976806641 x2_input_acceleration
   + 0.436536431312561 x3_input_acceleration >= -0.1640349579802753
 CBF_Constraint_sphere_right: 0.1853682994842529 x1_input_acceleration
   - 1.212094068527222 x2_input_acceleration
   + 0.2365366220474243 x3_input_acceleration >= -0.3592457527455407
 Relaxed_CLF_constraint: - 0.0213155150413513 x1_input_acceleration
   - 0.3170468807220459 x2_input_acceleration
   - 0.0207318067550659 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.187693178653717 x1_input_acceleration
   - 0.3596997261047363 x2_input_acceleration
   + 0.2394052743911743 x3_input_acceleration >= -0.024856876919818
 CBF_Constraint_sphere_low: 0.0876933932304382 x1_input_acceleration
   - 0.7596993446350098 x2_input_acceleration
   + 0.439405083656311 x3_input_acceleration >= -0.1638525132495353
 CBF_Constraint_sphere_right: 0.1876933574676514 x1_input_acceleration
   - 1.209698915481567 x2_input_acceleration
   + 0.2394052743911743 x3_input_acceleration >= -0.3583537869902683
 Relaxed_CLF_constraint: - 0.0201529860496521 x1_input_acceleration
   - 0.3158493041992188 x2_input_acceleration
   - 0.0192974805831909 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.1900584697723389 x1_input_acceleration
   - 0.3572971820831299 x2_input_acceleration
   + 0.2420862913131714 x3_input_acceleration >= -0.0249723176746285
 CBF_Constraint_sphere_low: 0.0900586843490601 x1_input_acceleration
   - 0.7572968006134033 x2_input_acceleration
   + 0.4420861005783081 x3_input_acceleration >= -0.1636372827926286
 CBF_Constraint_sphere_right: 0.1900586485862732 x1_input_acceleration
   - 1.207296371459961 x2_input_acceleration
   + 0.2420862913131714 x3_input_acceleration >= -0.3574481477211485
 Relaxed_CLF_constraint: - 0.0189703404903412 x1_input_acceleration
   - 0.3146480321884155 x2_input_acceleration
   - 0.0179569721221924 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.1924707889556885 x1_input_acceleration
   - 0.3548927307128906 x2_input_acceleration
   + 0.2446812391281128 x3_input_acceleration >= -0.0250906909238366
 CBF_Constraint_sphere_low: 0.0924710035324097 x1_input_acceleration
   - 0.7548923492431641 x2_input_acceleration
   + 0.4446810483932495 x3_input_acceleration >= -0.1634136450600684
 CBF_Constraint_sphere_right: 0.1924709677696228 x1_input_acceleration
   - 1.204891920089722 x2_input_acceleration
   + 0.2446812391281128 x3_input_acceleration >= -0.356544630328235
 Relaxed_CLF_constraint: - 0.0177641808986664 x1_input_acceleration
   - 0.3134458065032959 x2_input_acceleration
   - 0.0166594982147217 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.1949164867401123 x1_input_acceleration
   - 0.3524880409240723 x2_input_acceleration
   + 0.2472317218780518 x3_input_acceleration >= -0.0252159450250963
 CBF_Constraint_sphere_low: 0.0949167013168335 x1_input_acceleration
   - 0.7524876594543457 x2_input_acceleration
   + 0.4472315311431885 x3_input_acceleration >= -0.1631907250671573
 CBF_Constraint_sphere_right: 0.1949166655540466 x1_input_acceleration
   - 1.202487230300903 x2_input_acceleration
   + 0.2472317218780518 x3_input_acceleration >= -0.355647892462558
 Relaxed_CLF_constraint: - 0.0165413320064545 x1_input_acceleration
   - 0.3122434616088867 x2_input_acceleration
   - 0.0153842568397522 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.1972975134849548 x1_input_acceleration
   - 0.3500754833221436 x2_input_acceleration
   + 0.2496719360351562 x3_input_acceleration >= -0.0253288071235304
 CBF_Constraint_sphere_low: 0.097297728061676 x1_input_acceleration
   - 0.750075101852417 x2_input_acceleration
   + 0.449671745300293 x3_input_acceleration >= -0.1629460462065723
 CBF_Constraint_sphere_right: 0.1972976922988892 x1_input_acceleration
   - 1.200074672698975 x2_input_acceleration
   + 0.2496719360351562 x3_input_acceleration >= -0.3547354187708903
 Relaxed_CLF_constraint: - 0.0153508186340332 x1_input_acceleration
   - 0.3110371828079224 x2_input_acceleration
   - 0.0141641497612 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.1996918320655823 x1_input_acceleration
   - 0.3476629257202148 x2_input_acceleration
   + 0.2520989179611206 x3_input_acceleration >= -0.025450050537804
 CBF_Constraint_sphere_low: 0.0996920466423035 x1_input_acceleration
   - 0.7476625442504883 x2_input_acceleration
   + 0.4520987272262573 x3_input_acceleration >= -0.1627077608496116
 CBF_Constraint_sphere_right: 0.1996920108795166 x1_input_acceleration
   - 1.197662115097046 x2_input_acceleration
   + 0.2520989179611206 x3_input_acceleration >= -0.3538313263962506
 Relaxed_CLF_constraint: - 0.0141536593437195 x1_input_acceleration
   - 0.309830904006958 x2_input_acceleration
   - 0.0129506587982178 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2020013928413391 x1_input_acceleration
   - 0.3452434539794922 x2_input_acceleration
   + 0.2544254064559937 x3_input_acceleration >= -0.0255574731689571
 CBF_Constraint_sphere_low: 0.1020016074180603 x1_input_acceleration
   - 0.7452430725097656 x2_input_acceleration
   + 0.4544252157211304 x3_input_acceleration >= -0.162448460430715
 CBF_Constraint_sphere_right: 0.2020015716552734 x1_input_acceleration
   - 1.195242643356323 x2_input_acceleration
   + 0.2544254064559937 x3_input_acceleration >= -0.3529104747247273
 Relaxed_CLF_constraint: - 0.0129988789558411 x1_input_acceleration
   - 0.3086211681365967 x2_input_acceleration
   - 0.0117874145507812 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2043154835700989 x1_input_acceleration
   - 0.3428239822387695 x2_input_acceleration
   + 0.2567472457885742 x3_input_acceleration >= -0.0256730619611375
 CBF_Constraint_sphere_low: 0.1043156981468201 x1_input_acceleration
   - 0.742823600769043 x2_input_acceleration
   + 0.4567470550537109 x3_input_acceleration >= -0.1621966347598957
 CBF_Constraint_sphere_right: 0.2043156623840332 x1_input_acceleration
   - 1.192823171615601 x2_input_acceleration
   + 0.2567472457885742 x3_input_acceleration >= -0.3519977892146363
 Relaxed_CLF_constraint: - 0.0118418335914612 x1_input_acceleration
   - 0.3074114322662354 x2_input_acceleration
   - 0.010626494884491 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.206540048122406 x1_input_acceleration
   - 0.3403987884521484 x2_input_acceleration
   + 0.2589740753173828 x3_input_acceleration >= -0.0257744245861474
 CBF_Constraint_sphere_low: 0.1065402626991272 x1_input_acceleration
   - 0.7403984069824219 x2_input_acceleration
   + 0.4589738845825195 x3_input_acceleration >= -0.1619244138417188
 CBF_Constraint_sphere_right: 0.2065402269363403 x1_input_acceleration
   - 1.190397977828979 x2_input_acceleration
   + 0.2589740753173828 x3_input_acceleration >= -0.351068445662183
 Relaxed_CLF_constraint: - 0.0107295513153076 x1_input_acceleration
   - 0.3061988353729248 x2_input_acceleration
   - 0.00951308012008667 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2087661623954773 x1_input_acceleration
   - 0.3379735946655273 x2_input_acceleration
   + 0.2611993551254272 x3_input_acceleration >= -0.025883641092603
 CBF_Constraint_sphere_low: 0.1087663769721985 x1_input_acceleration
   - 0.7379732131958008 x2_input_acceleration
   + 0.461199164390564 x3_input_acceleration >= -0.1616598143471869
 CBF_Constraint_sphere_right: 0.2087663412094116 x1_input_acceleration
   - 1.187972784042358 x2_input_acceleration
   + 0.2611993551254272 x3_input_acceleration >= -0.3501469559913138
 Relaxed_CLF_constraint: - 0.00961649417877197 x1_input_acceleration
   - 0.3049862384796143 x2_input_acceleration
   - 0.00840044021606445 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2109014391899109 x1_input_acceleration
   - 0.3355443477630615 x2_input_acceleration
   + 0.2633317708969116 x3_input_acceleration >= -0.0259782619829544
 CBF_Constraint_sphere_low: 0.1109016537666321 x1_input_acceleration
   - 0.735543966293335 x2_input_acceleration
   + 0.4633315801620483 x3_input_acceleration >= -0.1613750640835413
 CBF_Constraint_sphere_right: 0.2109016180038452 x1_input_acceleration
   - 1.185543537139893 x2_input_acceleration
   + 0.2633317708969116 x3_input_acceleration >= -0.3492091481236278
 Relaxed_CLF_constraint: - 0.00854885578155518 x1_input_acceleration
   - 0.3037716150283813 x2_input_acceleration
   - 0.00733423233032227 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2130371928215027 x1_input_acceleration
   - 0.3331151008605957 x2_input_acceleration
   + 0.2654637098312378 x3_input_acceleration >= -0.0260803742959986
 CBF_Constraint_sphere_low: 0.1130374073982239 x1_input_acceleration
   - 0.7331147193908691 x2_input_acceleration
   + 0.4654635190963745 x3_input_acceleration >= -0.1610977337171115
 CBF_Constraint_sphere_right: 0.213037371635437 x1_input_acceleration
   - 1.183114290237427 x2_input_acceleration
   + 0.2654637098312378 x3_input_acceleration >= -0.3482788316786772
 Relaxed_CLF_constraint: - 0.00748097896575928 x1_input_acceleration
   - 0.3025569915771484 x2_input_acceleration
   - 0.00626826286315918 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2151731848716736 x1_input_acceleration
   - 0.3306858539581299 x2_input_acceleration
   + 0.2675951719284058 x3_input_acceleration >= -0.0261899523838075
 CBF_Constraint_sphere_low: 0.1151733994483948 x1_input_acceleration
   - 0.7306854724884033 x2_input_acceleration
   + 0.4675949811935425 x3_input_acceleration >= -0.1608278095208725
 CBF_Constraint_sphere_right: 0.2151733636856079 x1_input_acceleration
   - 1.180685043334961 x2_input_acceleration
   + 0.2675951719284058 x3_input_acceleration >= -0.3473559810085126
 Relaxed_CLF_constraint: - 0.00641298294067383 x1_input_acceleration
   - 0.3013423681259155 x2_input_acceleration
   - 0.0052025318145752 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2172181010246277 x1_input_acceleration
   - 0.328254222869873 x2_input_acceleration
   + 0.2696349620819092 x3_input_acceleration >= -0.0262843877553906
 CBF_Constraint_sphere_low: 0.1172183156013489 x1_input_acceleration
   - 0.7282538414001465 x2_input_acceleration
   + 0.4696347713470459 x3_input_acceleration >= -0.1605376523711698
 CBF_Constraint_sphere_right: 0.217218279838562 x1_input_acceleration
   - 1.178253412246704 x2_input_acceleration
   + 0.2696349620819092 x3_input_acceleration >= -0.3464169743359845
 Relaxed_CLF_constraint: - 0.00539052486419678 x1_input_acceleration
   - 0.3001265525817871 x2_input_acceleration
   - 0.00418263673782349 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2192630171775818 x1_input_acceleration
   - 0.3258225917816162 x2_input_acceleration
   + 0.271674633026123 x3_input_acceleration >= -0.0263859345617462
 CBF_Constraint_sphere_low: 0.119263231754303 x1_input_acceleration
   - 0.7258222103118896 x2_input_acceleration
   + 0.4716744422912598 x3_input_acceleration >= -0.1602545947353221
 CBF_Constraint_sphere_right: 0.2192631959915161 x1_input_acceleration
   - 1.175821781158447 x2_input_acceleration
   + 0.271674633026123 x3_input_acceleration >= -0.345485079098229
 Relaxed_CLF_constraint: - 0.00436806678771973 x1_input_acceleration
   - 0.2989107370376587 x2_input_acceleration
   - 0.00316280126571655 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2212170958518982 x1_input_acceleration
   - 0.3233897686004639 x2_input_acceleration
   + 0.273622989654541 x3_input_acceleration >= -0.0264718716000246
 CBF_Constraint_sphere_low: 0.1212173104286194 x1_input_acceleration
   - 0.7233893871307373 x2_input_acceleration
   + 0.4736227989196777 x3_input_acceleration >= -0.1599510993543604
 CBF_Constraint_sphere_right: 0.2212172746658325 x1_input_acceleration
   - 1.173388957977295 x2_input_acceleration
   + 0.273622989654541 x3_input_acceleration >= -0.3445370674452773
 Relaxed_CLF_constraint: - 0.00339102745056152 x1_input_acceleration
   - 0.2976943254470825 x2_input_acceleration
   - 0.00218862295150757 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2231711745262146 x1_input_acceleration
   - 0.3209569454193115 x2_input_acceleration
   + 0.275571346282959 x3_input_acceleration >= -0.0265645752111269
 CBF_Constraint_sphere_low: 0.1231713891029358 x1_input_acceleration
   - 0.720956563949585 x2_input_acceleration
   + 0.4755711555480957 x3_input_acceleration >= -0.1596543705462225
 CBF_Constraint_sphere_right: 0.2231713533401489 x1_input_acceleration
   - 1.170956134796143 x2_input_acceleration
   + 0.275571346282959 x3_input_acceleration >= -0.3435958223651495
 Relaxed_CLF_constraint: - 0.00241398811340332 x1_input_acceleration
   - 0.2964779138565063 x2_input_acceleration
   - 0.00121444463729858 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.225125253200531 x1_input_acceleration
   - 0.3185241222381592 x2_input_acceleration
   + 0.277519702911377 x3_input_acceleration >= -0.026664045395053
 CBF_Constraint_sphere_low: 0.1251254677772522 x1_input_acceleration
   - 0.7185237407684326 x2_input_acceleration
   + 0.4775195121765137 x3_input_acceleration >= -0.1593644083109084
 CBF_Constraint_sphere_right: 0.2251254320144653 x1_input_acceleration
   - 1.16852331161499 x2_input_acceleration
   + 0.277519702911377 x3_input_acceleration >= -0.3426613438578455
 Relaxed_CLF_constraint: - 0.00143694877624512 x1_input_acceleration
   - 0.2952615022659302 x2_input_acceleration
   - 2.4026632308959961e-04 x3_input_acceleration + relaxation_CLF
   <= -20.8
 CBF_Constraint_sphere_left: 0.2269886136054993 x1_input_acceleration
   - 0.316091775894165 x2_input_acceleration
   + 0.2793772220611572 x3_input_acceleration >= -0.0267473684252707
 CBF_Constraint_sphere_low: 0.1269888281822205 x1_input_acceleration
   - 0.7160913944244385 x2_input_acceleration
   + 0.4793770313262939 x3_input_acceleration >= -0.1590538464537602
 CBF_Constraint_sphere_right: 0.2269887924194336 x1_input_acceleration
   - 1.166090965270996 x2_input_acceleration
   + 0.2793772220611572 x3_input_acceleration >= -0.3417109208443213
 Relaxed_CLF_constraint: - 5.0526857376098633e-04 x1_input_acceleration
   - 0.2940453290939331 x2_input_acceleration
   + 6.8849325180053711e-04 x3_input_acceleration + relaxation_CLF
   <= -20.8
 CBF_Constraint_sphere_left: 0.2288520932197571 x1_input_acceleration
   - 0.3136594295501709 x2_input_acceleration
   + 0.2812347412109375 x3_input_acceleration >= -0.0268371244951965
 CBF_Constraint_sphere_low: 0.1288523077964783 x1_input_acceleration
   - 0.7136590480804443 x2_input_acceleration
   + 0.4812345504760742 x3_input_acceleration >= -0.1587497116758684
 CBF_Constraint_sphere_right: 0.2288522720336914 x1_input_acceleration
   - 1.163658618927002 x2_input_acceleration
   + 0.2812347412109375 x3_input_acceleration >= -0.3407669308705158
 Relaxed_CLF_constraint: 4.2647123336791992e-04 x1_input_acceleration
   - 0.292829155921936 x2_input_acceleration
   + 0.00161725282669067 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.230625331401825 x1_input_acceleration
   - 0.3112287521362305 x2_input_acceleration
   + 0.2830017805099487 x3_input_acceleration >= -0.0269103468530695
 CBF_Constraint_sphere_low: 0.1306255459785461 x1_input_acceleration
   - 0.7112283706665039 x2_input_acceleration
   + 0.4830015897750854 x3_input_acceleration >= -0.1584248410570958
 CBF_Constraint_sphere_right: 0.2306255102157593 x1_input_acceleration
   - 1.161227941513062 x2_input_acceleration
   + 0.2830017805099487 x3_input_acceleration >= -0.3398071164711857
 Relaxed_CLF_constraint: 0.00131309032440186 x1_input_acceleration
   - 0.2916138172149658 x2_input_acceleration
   + 0.00250077247619629 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2323984503746033 x1_input_acceleration
   - 0.30879807472229 x2_input_acceleration
   + 0.28476881980896 x3_input_acceleration >= -0.0269896428560245
 CBF_Constraint_sphere_low: 0.1323986649513245 x1_input_acceleration
   - 0.7087976932525635 x2_input_acceleration
   + 0.4847686290740967 x3_input_acceleration >= -0.1581060500438568
 CBF_Constraint_sphere_right: 0.2323986291885376 x1_input_acceleration
   - 1.158797264099121 x2_input_acceleration
   + 0.28476881980896 x3_input_acceleration >= -0.3388533757169269
 Relaxed_CLF_constraint: 0.00219964981079102 x1_input_acceleration
   - 0.2903984785079956 x2_input_acceleration
   + 0.0033842921257019 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2341715693473816 x1_input_acceleration
   - 0.3063673973083496 x2_input_acceleration
   + 0.2865358591079712 x3_input_acceleration >= -0.0270750261447127
 CBF_Constraint_sphere_low: 0.1341717839241028 x1_input_acceleration
   - 0.706367015838623 x2_input_acceleration
   + 0.4865356683731079 x3_input_acceleration >= -0.1577933463163511
 CBF_Constraint_sphere_right: 0.2341717481613159 x1_input_acceleration
   - 1.156366586685181 x2_input_acceleration
   + 0.2865358591079712 x3_input_acceleration >= -0.3379057222484013
 Relaxed_CLF_constraint: 0.00308620929718018 x1_input_acceleration
   - 0.2891831398010254 x2_input_acceleration
   + 0.00426781177520752 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2358549237251282 x1_input_acceleration
   - 0.3039400577545166 x2_input_acceleration
   + 0.2882128953933716 x3_input_acceleration >= -0.0271434442060589
 CBF_Constraint_sphere_low: 0.1358551383018494 x1_input_acceleration
   - 0.70393967628479 x2_input_acceleration
   + 0.4882127046585083 x3_input_acceleration >= -0.1574598328602314
 CBF_Constraint_sphere_right: 0.2358551025390625 x1_input_acceleration
   - 1.153939247131348 x2_input_acceleration
   + 0.2882128953933716 x3_input_acceleration >= -0.3369425221337009
 Relaxed_CLF_constraint: 0.00392788648605347 x1_input_acceleration
   - 0.2879694700241089 x2_input_acceleration
   + 0.00510633 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2375382781028748 x1_input_acceleration
   - 0.3015127182006836 x2_input_acceleration
   + 0.289889931678772 x3_input_acceleration >= -0.0272176313223917
 CBF_Constraint_sphere_low: 0.1375384926795959 x1_input_acceleration
   - 0.701512336730957 x2_input_acceleration
   + 0.4898897409439087 x3_input_acceleration >= -0.1571320884590982
 CBF_Constraint_sphere_right: 0.2375384569168091 x1_input_acceleration
   - 1.151511907577515 x2_input_acceleration
   + 0.289889931678772 x3_input_acceleration >= -0.3359850910739871
 Relaxed_CLF_constraint: 0.00476956367492676 x1_input_acceleration
   - 0.2867558002471924 x2_input_acceleration
   + 0.00594484806060791 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2392216324806213 x1_input_acceleration
   - 0.2990853786468506 x2_input_acceleration
   + 0.2915668487548828 x3_input_acceleration >= -0.027297570114969
 CBF_Constraint_sphere_low: 0.1392218470573425 x1_input_acceleration
   - 0.699084997177124 x2_input_acceleration
   + 0.4915666580200195 x3_input_acceleration >= -0.1568100838132921
 CBF_Constraint_sphere_right: 0.2392218112945557 x1_input_acceleration
   - 1.149084568023682 x2_input_acceleration
   + 0.2915668487548828 x3_input_acceleration >= -0.3350334116905179
 Relaxed_CLF_constraint: 0.00561124086380005 x1_input_acceleration
   - 0.2855421304702759 x2_input_acceleration
   + 0.00678330659866333 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2408154606819153 x1_input_acceleration
   - 0.2966628074645996 x2_input_acceleration
   + 0.2931541204452515 x3_input_acceleration >= -0.0273600614425626
 CBF_Constraint_sphere_low: 0.1408156752586365 x1_input_acceleration
   - 0.696662425994873 x2_input_acceleration
   + 0.4931539297103882 x3_input_acceleration >= -0.1564670971451016
 CBF_Constraint_sphere_right: 0.2408156394958496 x1_input_acceleration
   - 1.146661996841431 x2_input_acceleration
   + 0.2931541204452515 x3_input_acceleration >= -0.3340663113900503
 Relaxed_CLF_constraint: 0.00640815496444702 x1_input_acceleration
   - 0.2843308448791504 x2_input_acceleration
   + 0.00757694244384766 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2424092888832092 x1_input_acceleration
   - 0.2942402362823486 x2_input_acceleration
   + 0.2947415113449097 x3_input_acceleration >= -0.0274280346235593
 CBF_Constraint_sphere_low: 0.1424095034599304 x1_input_acceleration
   - 0.6942398548126221 x2_input_acceleration
   + 0.4947413206100464 x3_input_acceleration >= -0.1561296042512318
 CBF_Constraint_sphere_right: 0.2424094676971436 x1_input_acceleration
   - 1.14423942565918 x2_input_acceleration
   + 0.2947415113449097 x3_input_acceleration >= -0.3331046929429858
 Relaxed_CLF_constraint: 0.00720506906509399 x1_input_acceleration
   - 0.2831195592880249 x2_input_acceleration
   + 0.00837063789367676 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2440031170845032 x1_input_acceleration
   - 0.2918176651000977 x2_input_acceleration
   + 0.2963287830352783 x3_input_acceleration >= -0.0275014546166489
 CBF_Constraint_sphere_low: 0.1440033316612244 x1_input_acceleration
   - 0.6918172836303711 x2_input_acceleration
   + 0.496328592300415 x3_input_acceleration >= -0.1557975462485374
 CBF_Constraint_sphere_right: 0.2440032958984375 x1_input_acceleration
   - 1.141816854476929 x2_input_acceleration
   + 0.2963287830352783 x3_input_acceleration >= -0.3321485213080143
 Relaxed_CLF_constraint: 0.00800198316574097 x1_input_acceleration
   - 0.2819082736968994 x2_input_acceleration
   + 0.00916427373886108 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2455077767372131 x1_input_acceleration
   - 0.2894015312194824 x2_input_acceleration
   + 0.297826886177063 x3_input_acceleration >= -0.0275570422101389
 CBF_Constraint_sphere_low: 0.1455079913139343 x1_input_acceleration
   - 0.6894011497497559 x2_input_acceleration
   + 0.4978266954421997 x3_input_acceleration >= -0.1554444848768505
 CBF_Constraint_sphere_right: 0.2455079555511475 x1_input_acceleration
   - 1.139400720596313 x2_input_acceleration
   + 0.297826886177063 x3_input_acceleration >= -0.3311772531160569
 Relaxed_CLF_constraint: 0.00875431299209595 x1_input_acceleration
   - 0.2807002067565918 x2_input_acceleration
   + 0.00991332530975342 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2470124363899231 x1_input_acceleration
   - 0.2869853973388672 x2_input_acceleration
   + 0.2993248701095581 x3_input_acceleration >= -0.0276177849707843
 CBF_Constraint_sphere_low: 0.1470126509666443 x1_input_acceleration
   - 0.6869850158691406 x2_input_acceleration
   + 0.4993246793746948 x3_input_acceleration >= -0.1550965667514014
 CBF_Constraint_sphere_right: 0.2470126152038574 x1_input_acceleration
   - 1.136984586715698 x2_input_acceleration
   + 0.2993248701095581 x3_input_acceleration >= -0.330211140091255
 Relaxed_CLF_constraint: 0.00950664281845093 x1_input_acceleration
   - 0.2794921398162842 x2_input_acceleration
   + 0.010662317276001 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2485170960426331 x1_input_acceleration
   - 0.284569263458252 x2_input_acceleration
   + 0.3008228540420532 x3_input_acceleration >= -0.0276837005611604
 CBF_Constraint_sphere_low: 0.1485173106193542 x1_input_acceleration
   - 0.6845688819885254 x2_input_acceleration
   + 0.5008226633071899 x3_input_acceleration >= -0.1547538214556831
 CBF_Constraint_sphere_right: 0.2485172748565674 x1_input_acceleration
   - 1.134568452835083 x2_input_acceleration
   + 0.3008228540420532 x3_input_acceleration >= -0.3292501998961838
 Relaxed_CLF_constraint: 0.0102589726448059 x1_input_acceleration
   - 0.2782840728759766 x2_input_acceleration
   + 0.0114113092422485 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2499330639839172 x1_input_acceleration
   - 0.282160758972168 x2_input_acceleration
   + 0.302232027053833 x3_input_acceleration >= -0.0277313571383019
 CBF_Constraint_sphere_low: 0.1499332785606384 x1_input_acceleration
   - 0.6821603775024414 x2_input_acceleration
   + 0.5022318363189697 x3_input_acceleration >= -0.154389896516635
 CBF_Constraint_sphere_right: 0.2499332427978516 x1_input_acceleration
   - 1.132159948348999 x2_input_acceleration
   + 0.302232027053833 x3_input_acceleration >= -0.3282742431695317
 Relaxed_CLF_constraint: 0.010966956615448 x1_input_acceleration
   - 0.2770798206329346 x2_input_acceleration
   + 0.0121158957481384 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2513490319252014 x1_input_acceleration
   - 0.279752254486084 x2_input_acceleration
   + 0.3036412000656128 x3_input_acceleration >= -0.027783909529267
 CBF_Constraint_sphere_low: 0.1513492465019226 x1_input_acceleration
   - 0.6797518730163574 x2_input_acceleration
   + 0.5036410093307495 x3_input_acceleration >= -0.1540308673914105
 CBF_Constraint_sphere_right: 0.2513492107391357 x1_input_acceleration
   - 1.129751443862915 x2_input_acceleration
   + 0.3036412000656128 x3_input_acceleration >= -0.3273031822567033
 Relaxed_CLF_constraint: 0.0116749405860901 x1_input_acceleration
   - 0.2758755683898926 x2_input_acceleration
   + 0.0128204822540283 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2527649998664856 x1_input_acceleration
   - 0.27734375 x2_input_acceleration
   + 0.3050503730773926 x3_input_acceleration >= -0.0278413577340558
 CBF_Constraint_sphere_low: 0.1527652144432068 x1_input_acceleration
   - 0.6773433685302734 x2_input_acceleration
   + 0.5050501823425293 x3_input_acceleration >= -0.1536767340800097
 CBF_Constraint_sphere_right: 0.2527651786804199 x1_input_acceleration
   - 1.127342939376831 x2_input_acceleration
   + 0.3050503730773926 x3_input_acceleration >= -0.3263370171576986
 Relaxed_CLF_constraint: 0.0123829245567322 x1_input_acceleration
   - 0.2746713161468506 x2_input_acceleration
   + 0.0135250687599182 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2540926337242126 x1_input_acceleration
   - 0.274944543838501 x2_input_acceleration
   + 0.3063709735870361 x3_input_acceleration >= -0.0278801855390342
 CBF_Constraint_sphere_low: 0.1540928483009338 x1_input_acceleration
   - 0.6749441623687744 x2_input_acceleration
   + 0.5063707828521729 x3_input_acceleration >= -0.1533013994848759
 CBF_Constraint_sphere_right: 0.254092812538147 x1_input_acceleration
   - 1.124943733215332 x2_input_acceleration
   + 0.3063709735870361 x3_input_acceleration >= -0.3253561834351656
 Relaxed_CLF_constraint: 0.0130467414855957 x1_input_acceleration
   - 0.2734717130661011 x2_input_acceleration
   + 0.01418536901474 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2554201483726501 x1_input_acceleration
   - 0.272545337677002 x2_input_acceleration
   + 0.3076915740966797 x3_input_acceleration >= -0.0279236295135675
 CBF_Constraint_sphere_low: 0.1554203629493713 x1_input_acceleration
   - 0.6725449562072754 x2_input_acceleration
   + 0.5076913833618164 x3_input_acceleration >= -0.1529306870197488
 CBF_Constraint_sphere_right: 0.2554203271865845 x1_input_acceleration
   - 1.122544527053833 x2_input_acceleration
   + 0.3076915740966797 x3_input_acceleration >= -0.324379965882177
 Relaxed_CLF_constraint: 0.0137104988098145 x1_input_acceleration
   - 0.2722721099853516 x2_input_acceleration
   + 0.0148456692695618 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2567477822303772 x1_input_acceleration
   - 0.2701461315155029 x2_input_acceleration
   + 0.3090121746063232 x3_input_acceleration >= -0.0279717200269844
 CBF_Constraint_sphere_low: 0.1567479968070984 x1_input_acceleration
   - 0.6701457500457764 x2_input_acceleration
   + 0.50901198387146 x3_input_acceleration >= -0.1525646151330534
 CBF_Constraint_sphere_right: 0.2567479610443115 x1_input_acceleration
   - 1.120145320892334 x2_input_acceleration
   + 0.3090121746063232 x3_input_acceleration >= -0.3234083948680825
 Relaxed_CLF_constraint: 0.014374315738678 x1_input_acceleration
   - 0.2710725069046021 x2_input_acceleration
   + 0.0155059695243835 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2579874396324158 x1_input_acceleration
   - 0.2677576541900635 x2_input_acceleration
   + 0.3102447986602783 x3_input_acceleration >= -0.0280008788703029
 CBF_Constraint_sphere_low: 0.157987654209137 x1_input_acceleration
   - 0.6677572727203369 x2_input_acceleration
   + 0.510244607925415 x3_input_acceleration >= -0.1521773585175922
 CBF_Constraint_sphere_right: 0.2579876184463501 x1_input_acceleration
   - 1.117756843566895 x2_input_acceleration
   + 0.3102447986602783 x3_input_acceleration >= -0.3224224519270008
 Relaxed_CLF_constraint: 0.0149941444396973 x1_input_acceleration
   - 0.2698782682418823 x2_input_acceleration
   + 0.0161222815513611 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2592269778251648 x1_input_acceleration
   - 0.265369176864624 x2_input_acceleration
   + 0.3114773035049438 x3_input_acceleration >= -0.0280343841652219
 CBF_Constraint_sphere_low: 0.159227192401886 x1_input_acceleration
   - 0.6653687953948975 x2_input_acceleration
   + 0.5114771127700806 x3_input_acceleration >= -0.1517944423932655
 CBF_Constraint_sphere_right: 0.2592271566390991 x1_input_acceleration
   - 1.115368366241455 x2_input_acceleration
   + 0.3114773035049438 x3_input_acceleration >= -0.3214408554375089
 Relaxed_CLF_constraint: 0.0156139135360718 x1_input_acceleration
   - 0.2686840295791626 x2_input_acceleration
   + 0.0167385339736938 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2604666352272034 x1_input_acceleration
   - 0.2629806995391846 x2_input_acceleration
   + 0.3127098083496094 x3_input_acceleration >= -0.0280722851586873
 CBF_Constraint_sphere_low: 0.1604668498039246 x1_input_acceleration
   - 0.662980318069458 x2_input_acceleration
   + 0.5127096176147461 x3_input_acceleration >= -0.1514159160070336
 CBF_Constraint_sphere_right: 0.2604668140411377 x1_input_acceleration
   - 1.112979888916016 x2_input_acceleration
   + 0.3127098083496094 x3_input_acceleration >= -0.3204636546465741
 Relaxed_CLF_constraint: 0.0162337422370911 x1_input_acceleration
   - 0.2674897909164429 x2_input_acceleration
   + 0.0173547863960266 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2617061734199524 x1_input_acceleration
   - 0.2605922222137451 x2_input_acceleration
   + 0.3139423131942749 x3_input_acceleration >= -0.0281145508745461
 CBF_Constraint_sphere_low: 0.1617063879966736 x1_input_acceleration
   - 0.6605918407440186 x2_input_acceleration
   + 0.5139421224594116 x3_input_acceleration >= -0.1510417603036466
 CBF_Constraint_sphere_right: 0.2617063522338867 x1_input_acceleration
   - 1.110591411590576 x2_input_acceleration
   + 0.3139423131942749 x3_input_acceleration >= -0.319490818578022
 Relaxed_CLF_constraint: 0.0168535113334656 x1_input_acceleration
   - 0.2662955522537231 x2_input_acceleration
   + 0.0179710388183594 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2628582119941711 x1_input_acceleration
   - 0.2582159042358398 x2_input_acceleration
   + 0.3150870800018311 x3_input_acceleration >= -0.0281374401992963
 CBF_Constraint_sphere_low: 0.1628584265708923 x1_input_acceleration
   - 0.6582155227661133 x2_input_acceleration
   + 0.5150868892669678 x3_input_acceleration >= -0.150646261252534
 CBF_Constraint_sphere_right: 0.2628583908081055 x1_input_acceleration
   - 1.108215093612671 x2_input_acceleration
   + 0.3150870800018311 x3_input_acceleration >= -0.318503773828312
 Relaxed_CLF_constraint: 0.017429530620575 x1_input_acceleration
   - 0.2651073932647705 x2_input_acceleration
   + 0.0185434222221375 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2640102505683899 x1_input_acceleration
   - 0.2558395862579346 x2_input_acceleration
   + 0.3162318468093872 x3_input_acceleration >= -0.0281644718095727
 CBF_Constraint_sphere_low: 0.1640104651451111 x1_input_acceleration
   - 0.655839204788208 x2_input_acceleration
   + 0.5162316560745239 x3_input_acceleration >= -0.1502549044869476
 CBF_Constraint_sphere_right: 0.2640104293823242 x1_input_acceleration
   - 1.105838775634766 x2_input_acceleration
   + 0.3162318468093872 x3_input_acceleration >= -0.3175208713641281
 Relaxed_CLF_constraint: 0.0180055499076843 x1_input_acceleration
   - 0.2639192342758179 x2_input_acceleration
   + 0.0191158056259155 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2651622891426086 x1_input_acceleration
   - 0.2534632682800293 x2_input_acceleration
   + 0.3173766136169434 x3_input_acceleration >= -0.0281956457053753
 CBF_Constraint_sphere_low: 0.1651625037193298 x1_input_acceleration
   - 0.6534628868103027 x2_input_acceleration
   + 0.5173764228820801 x3_input_acceleration >= -0.1498676900068873
 CBF_Constraint_sphere_right: 0.265162467956543 x1_input_acceleration
   - 1.10346245765686 x2_input_acceleration
   + 0.3173766136169434 x3_input_acceleration >= -0.3165421111854704
 Relaxed_CLF_constraint: 0.0185815691947937 x1_input_acceleration
   - 0.2627310752868652 x2_input_acceleration
   + 0.0196881890296936 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2662270665168762 x1_input_acceleration
   - 0.2511005401611328 x2_input_acceleration
   + 0.318434 x3_input_acceleration >= -0.0282071361452779
 CBF_Constraint_sphere_low: 0.1662272810935974 x1_input_acceleration
   - 0.6511001586914062 x2_input_acceleration
   + 0.5184338092803955 x3_input_acceleration >= -0.1494591350581815
 CBF_Constraint_sphere_right: 0.2662272453308105 x1_input_acceleration
   - 1.101099729537964 x2_input_acceleration
   + 0.318434 x3_input_acceleration >= -0.3155494432276816
 Relaxed_CLF_constraint: 0.0191139578819275 x1_input_acceleration
   - 0.261549711227417 x2_input_acceleration
   + 0.0202168822288513 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2672918438911438 x1_input_acceleration
   - 0.2487378120422363 x2_input_acceleration
   + 0.3194915056228638 x3_input_acceleration >= -0.0282225627788627
 CBF_Constraint_sphere_low: 0.167292058467865 x1_input_acceleration
   - 0.6487374305725098 x2_input_acceleration
   + 0.5194913148880005 x3_input_acceleration >= -0.1490545282240753
 CBF_Constraint_sphere_right: 0.2672920227050781 x1_input_acceleration
   - 1.098737001419067 x2_input_acceleration
   + 0.3194915056228638 x3_input_acceleration >= -0.314560711463575
 Relaxed_CLF_constraint: 0.0196463465690613 x1_input_acceleration
   - 0.2603683471679688 x2_input_acceleration
   + 0.0207456350326538 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2683566212654114 x1_input_acceleration
   - 0.2463750839233398 x2_input_acceleration
   + 0.3205488920211792 x3_input_acceleration >= -0.0282418875828065
 CBF_Constraint_sphere_low: 0.1683568358421326 x1_input_acceleration
   - 0.6463747024536133 x2_input_acceleration
   + 0.5205487012863159 x3_input_acceleration >= -0.1486538076394105
 CBF_Constraint_sphere_right: 0.2683568000793457 x1_input_acceleration
   - 1.096374273300171 x2_input_acceleration
   + 0.3205488920211792 x3_input_acceleration >= -0.3135758778698273
 Relaxed_CLF_constraint: 0.0201787352561951 x1_input_acceleration
   - 0.2591869831085205 x2_input_acceleration
   + 0.0212743282318115 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.269421398639679 x1_input_acceleration
   - 0.2440123558044434 x2_input_acceleration
   + 0.3216062784194946 x3_input_acceleration >= -0.0282651295372581
 CBF_Constraint_sphere_low: 0.1694216132164001 x1_input_acceleration
   - 0.6440119743347168 x2_input_acceleration
   + 0.5216060876846313 x3_input_acceleration >= -0.1482570042052537
 CBF_Constraint_sphere_right: 0.2694215774536133 x1_input_acceleration
   - 1.094011545181274 x2_input_acceleration
   + 0.3216062784194946 x3_input_acceleration >= -0.3125949614265876
 Relaxed_CLF_constraint: 0.0207111239433289 x1_input_acceleration
   - 0.2580056190490723 x2_input_acceleration
   + 0.0218030214309692 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2703993916511536 x1_input_acceleration
   - 0.2416648864746094 x2_input_acceleration
   + 0.3225767612457275 x3_input_acceleration >= -0.0282683788139707
 CBF_Constraint_sphere_low: 0.1703996062278748 x1_input_acceleration
   - 0.6416645050048828 x2_input_acceleration
   + 0.5225765705108643 x3_input_acceleration >= -0.1478389087081681
 CBF_Constraint_sphere_right: 0.2703995704650879 x1_input_acceleration
   - 1.09166407585144 x2_input_acceleration
   + 0.3225767612457275 x3_input_acceleration >= -0.3116005372770166
 Relaxed_CLF_constraint: 0.0212001204490662 x1_input_acceleration
   - 0.2568318843841553 x2_input_acceleration
   + 0.0222882628440857 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2713773846626282 x1_input_acceleration
   - 0.2393174171447754 x2_input_acceleration
   + 0.3235472440719604 x3_input_acceleration >= -0.0282753325504338
 CBF_Constraint_sphere_low: 0.1713775992393494 x1_input_acceleration
   - 0.6393170356750488 x2_input_acceleration
   + 0.5235470533370972 x3_input_acceleration >= -0.147424517670833
 CBF_Constraint_sphere_right: 0.2713775634765625 x1_input_acceleration
   - 1.089316606521606 x2_input_acceleration
   + 0.3235472440719604 x3_input_acceleration >= -0.3106098175871962
 Relaxed_CLF_constraint: 0.0216891169548035 x1_input_acceleration
   - 0.2556581497192383 x2_input_acceleration
   + 0.0227735042572021 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2723553776741028 x1_input_acceleration
   - 0.2369699478149414 x2_input_acceleration
   + 0.3245177268981934 x3_input_acceleration >= -0.0282859907466474
 CBF_Constraint_sphere_low: 0.172355592250824 x1_input_acceleration
   - 0.6369695663452148 x2_input_acceleration
   + 0.5245175361633301 x3_input_acceleration >= -0.1470138310932484
 CBF_Constraint_sphere_right: 0.2723555564880371 x1_input_acceleration
   - 1.086969137191772 x2_input_acceleration
   + 0.3245177268981934 x3_input_acceleration >= -0.3096228023571262
 Relaxed_CLF_constraint: 0.0221781134605408 x1_input_acceleration
   - 0.2544844150543213 x2_input_acceleration
   + 0.0232587456703186 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2733332514762878 x1_input_acceleration
   - 0.2346224784851074 x2_input_acceleration
   + 0.3254882097244263 x3_input_acceleration >= -0.0283003371106766
 CBF_Constraint_sphere_low: 0.173333466053009 x1_input_acceleration
   - 0.6346220970153809 x2_input_acceleration
   + 0.525488018989563 x3_input_acceleration >= -0.1466068386439311
 CBF_Constraint_sphere_right: 0.2733334302902222 x1_input_acceleration
   - 1.084621667861938 x2_input_acceleration
   + 0.3254882097244263 x3_input_acceleration >= -0.3086394752948612
 Relaxed_CLF_constraint: 0.0226670503616333 x1_input_acceleration
   - 0.2533106803894043 x2_input_acceleration
   + 0.0237439870834351 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2742246985435486 x1_input_acceleration
   - 0.2322914600372314 x2_input_acceleration
   + 0.3263721466064453 x3_input_acceleration >= -0.028294321444507
 CBF_Constraint_sphere_low: 0.1742249131202698 x1_input_acceleration
   - 0.6322910785675049 x2_input_acceleration
   + 0.526371955871582 x3_input_acceleration >= -0.1461784410789747
 CBF_Constraint_sphere_right: 0.2742248773574829 x1_input_acceleration
   - 1.082290649414062 x2_input_acceleration
   + 0.3263721466064453 x3_input_acceleration >= -0.3076427778128347
 Relaxed_CLF_constraint: 0.0231127738952637 x1_input_acceleration
   - 0.2521451711654663 x2_input_acceleration
   + 0.0241859555244446 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2751162648200989 x1_input_acceleration
   - 0.2299604415893555 x2_input_acceleration
   + 0.3272560834884644 x3_input_acceleration >= -0.0282918270111857
 CBF_Constraint_sphere_low: 0.1751164793968201 x1_input_acceleration
   - 0.6299600601196289 x2_input_acceleration
   + 0.5272558927536011 x3_input_acceleration >= -0.1457535587864151
 CBF_Constraint_sphere_right: 0.2751164436340332 x1_input_acceleration
   - 1.079959630966187 x2_input_acceleration
   + 0.3272560834884644 x3_input_acceleration >= -0.3066496015636672
 Relaxed_CLF_constraint: 0.0235585570335388 x1_input_acceleration
   - 0.2509796619415283 x2_input_acceleration
   + 0.0246279239654541 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2760077118873596 x1_input_acceleration
   - 0.2276294231414795 x2_input_acceleration
   + 0.3281400203704834 x3_input_acceleration >= -0.0282928210674399
 CBF_Constraint_sphere_low: 0.1760079264640808 x1_input_acceleration
   - 0.6276290416717529 x2_input_acceleration
   + 0.5281398296356201 x3_input_acceleration >= -0.1453321709438825
 CBF_Constraint_sphere_right: 0.2760078907012939 x1_input_acceleration
   - 1.077628612518311 x2_input_acceleration
   + 0.3281400203704834 x3_input_acceleration >= -0.3056599138040644
 Relaxed_CLF_constraint: 0.0240042805671692 x1_input_acceleration
   - 0.2498141527175903 x2_input_acceleration
   + 0.0250698924064636 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2768992781639099 x1_input_acceleration
   - 0.2252984046936035 x2_input_acceleration
   + 0.3290238380432129 x3_input_acceleration >= -0.0282973168514659
 CBF_Constraint_sphere_low: 0.1768994927406311 x1_input_acceleration
   - 0.625298023223877 x2_input_acceleration
   + 0.5290236473083496 x3_input_acceleration >= -0.1449142669477525
 CBF_Constraint_sphere_right: 0.2768994569778442 x1_input_acceleration
   - 1.075297594070435 x2_input_acceleration
   + 0.3290238380432129 x3_input_acceleration >= -0.3046737277722441
 Relaxed_CLF_constraint: 0.0244500637054443 x1_input_acceleration
   - 0.2486486434936523 x2_input_acceleration
   + 0.0255118012428284 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.277704656124115 x1_input_acceleration
   - 0.2229855060577393 x2_input_acceleration
   + 0.3298215866088867 x3_input_acceleration >= -0.0282811727345106
 CBF_Constraint_sphere_low: 0.1777048707008362 x1_input_acceleration
   - 0.6229851245880127 x2_input_acceleration
   + 0.5298213958740234 x3_input_acceleration >= -0.1444750495136603
 CBF_Constraint_sphere_right: 0.2777048349380493 x1_input_acceleration
   - 1.07298469543457 x2_input_acceleration
   + 0.3298215866088867 x3_input_acceleration >= -0.3036746027444975
 Relaxed_CLF_constraint: 0.0248527526855469 x1_input_acceleration
   - 0.2474921941757202 x2_input_acceleration
   + 0.0259106755256653 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2785101532936096 x1_input_acceleration
   - 0.220672607421875 x2_input_acceleration
   + 0.3306193351745605 x3_input_acceleration >= -0.0282683624863168
 CBF_Constraint_sphere_low: 0.1785103678703308 x1_input_acceleration
   - 0.6206722259521484 x2_input_acceleration
   + 0.5306191444396973 x3_input_acceleration >= -0.144039159987878
 CBF_Constraint_sphere_right: 0.2785103321075439 x1_input_acceleration
   - 1.070671796798706 x2_input_acceleration
   + 0.3306193351745605 x3_input_acceleration >= -0.3026788115855231
 Relaxed_CLF_constraint: 0.0252555012702942 x1_input_acceleration
   - 0.2463357448577881 x2_input_acceleration
   + 0.0263095498085022 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2793155312538147 x1_input_acceleration
   - 0.2183597087860107 x2_input_acceleration
   + 0.3314170837402344 x3_input_acceleration >= -0.0282588529538984
 CBF_Constraint_sphere_low: 0.1793157458305359 x1_input_acceleration
   - 0.6183593273162842 x2_input_acceleration
   + 0.5314168930053711 x3_input_acceleration >= -0.1436065771383227
 CBF_Constraint_sphere_right: 0.279315710067749 x1_input_acceleration
   - 1.068358898162842 x2_input_acceleration
   + 0.3314170837402344 x3_input_acceleration >= -0.3016863211423134
 Relaxed_CLF_constraint: 0.0256581902503967 x1_input_acceleration
   - 0.245179295539856 x2_input_acceleration
   + 0.0267084240913391 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2801209092140198 x1_input_acceleration
   - 0.2160468101501465 x2_input_acceleration
   + 0.3322147130966187 x3_input_acceleration >= -0.0282526408882028
 CBF_Constraint_sphere_low: 0.180121123790741 x1_input_acceleration
   - 0.6160464286804199 x2_input_acceleration
   + 0.5322145223617554 x3_input_acceleration >= -0.1431772798345727
 CBF_Constraint_sphere_right: 0.2801210880279541 x1_input_acceleration
   - 1.066045999526978 x2_input_acceleration
   + 0.3322147130966187 x3_input_acceleration >= -0.3006971281658265
 Relaxed_CLF_constraint: 0.0260608792304993 x1_input_acceleration
   - 0.2440228462219238 x2_input_acceleration
   + 0.0271072387695312 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2809032797813416 x1_input_acceleration
   - 0.2137391567230225 x2_input_acceleration
   + 0.3329893350601196 x3_input_acceleration >= -0.028243244243091
 CBF_Constraint_sphere_low: 0.1809034943580627 x1_input_acceleration
   - 0.6137387752532959 x2_input_acceleration
   + 0.5329891443252563 x3_input_acceleration >= -0.1427446966222356
 CBF_Constraint_sphere_right: 0.2809034585952759 x1_input_acceleration
   - 1.063738346099854 x2_input_acceleration
   + 0.3329893350601196 x3_input_acceleration >= -0.299706979819455
 Relaxed_CLF_constraint: 0.0264520645141602 x1_input_acceleration
   - 0.2428690195083618 x2_input_acceleration
   + 0.0274945497512817 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2816230654716492 x1_input_acceleration
   - 0.2114458084106445 x2_input_acceleration
   + 0.3337012529373169 x3_input_acceleration >= -0.0282193517780037
 CBF_Constraint_sphere_low: 0.1816232800483704 x1_input_acceleration
   - 0.611445426940918 x2_input_acceleration
   + 0.5337010622024536 x3_input_acceleration >= -0.1422973374446294
 CBF_Constraint_sphere_right: 0.2816232442855835 x1_input_acceleration
   - 1.061444997787476 x2_input_acceleration
   + 0.3337012529373169 x3_input_acceleration >= -0.2987084153154816
 Relaxed_CLF_constraint: 0.026811957359314 x1_input_acceleration
   - 0.2417223453521729 x2_input_acceleration
   + 0.0278505086898804 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2823428511619568 x1_input_acceleration
   - 0.2091524600982666 x2_input_acceleration
   + 0.3344131708145142 x3_input_acceleration >= -0.0281986014954093
 CBF_Constraint_sphere_low: 0.182343065738678 x1_input_acceleration
   - 0.60915207862854 x2_input_acceleration
   + 0.5344129800796509 x3_input_acceleration >= -0.1418531204495161
 CBF_Constraint_sphere_right: 0.2823430299758911 x1_input_acceleration
   - 1.059151649475098 x2_input_acceleration
   + 0.3344131708145142 x3_input_acceleration >= -0.2977129929940011
 Relaxed_CLF_constraint: 0.0271718502044678 x1_input_acceleration
   - 0.2405756711959839 x2_input_acceleration
   + 0.028206467628479 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2830625176429749 x1_input_acceleration
   - 0.2068591117858887 x2_input_acceleration
   + 0.3351250886917114 x3_input_acceleration >= -0.0281809765234634
 CBF_Constraint_sphere_low: 0.183062732219696 x1_input_acceleration
   - 0.6068587303161621 x2_input_acceleration
   + 0.5351248979568481 x3_input_acceleration >= -0.1414120347255029
 CBF_Constraint_sphere_right: 0.2830626964569092 x1_input_acceleration
   - 1.05685830116272 x2_input_acceleration
   + 0.3351250886917114 x3_input_acceleration >= -0.2967206959831584
 Relaxed_CLF_constraint: 0.0275316834449768 x1_input_acceleration
   - 0.2394289970397949 x2_input_acceleration
   + 0.0285624265670776 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2837821841239929 x1_input_acceleration
   - 0.2045657634735107 x2_input_acceleration
   + 0.3358370065689087 x3_input_acceleration >= -0.0281664936482123
 CBF_Constraint_sphere_low: 0.1837823987007141 x1_input_acceleration
   - 0.6045653820037842 x2_input_acceleration
   + 0.5358368158340454 x3_input_acceleration >= -0.1409740910981846
 CBF_Constraint_sphere_right: 0.2837823629379272 x1_input_acceleration
   - 1.054564952850342 x2_input_acceleration
   + 0.3358370065689087 x3_input_acceleration >= -0.2957315410690106
 Relaxed_CLF_constraint: 0.0278915166854858 x1_input_acceleration
   - 0.238282322883606 x2_input_acceleration
   + 0.0289183855056763 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2844166159629822 x1_input_acceleration
   - 0.2022933959960938 x2_input_acceleration
   + 0.3364635705947876 x3_input_acceleration >= -0.0281307909592151
 CBF_Constraint_sphere_low: 0.1844168305397034 x1_input_acceleration
   - 0.6022930145263672 x2_input_acceleration
   + 0.5364633798599243 x3_input_acceleration >= -0.1405148501660755
 CBF_Constraint_sphere_right: 0.2844167947769165 x1_input_acceleration
   - 1.052292585372925 x2_input_acceleration
   + 0.3364635705947876 x3_input_acceleration >= -0.2947300831798506
 Relaxed_CLF_constraint: 0.0282087326049805 x1_input_acceleration
   - 0.2371461391448975 x2_input_acceleration
   + 0.0292316675186157 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2850510478019714 x1_input_acceleration
   - 0.2000210285186768 x2_input_acceleration
   + 0.337090015411377 x3_input_acceleration >= -0.0280980475481784
 CBF_Constraint_sphere_low: 0.1850512623786926 x1_input_acceleration
   - 0.6000206470489502 x2_input_acceleration
   + 0.5370898246765137 x3_input_acceleration >= -0.1400585565910092
 CBF_Constraint_sphere_right: 0.2850512266159058 x1_input_acceleration
   - 1.050020217895508 x2_input_acceleration
   + 0.337090015411377 x3_input_acceleration >= -0.2937315845686511
 Relaxed_CLF_constraint: 0.0285259485244751 x1_input_acceleration
   - 0.236009955406189 x2_input_acceleration
   + 0.0295448899269104 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2856854796409607 x1_input_acceleration
   - 0.1977486610412598 x2_input_acceleration
   + 0.3377165794372559 x3_input_acceleration >= -0.0280683035620243
 CBF_Constraint_sphere_low: 0.1856856942176819 x1_input_acceleration
   - 0.5977482795715332 x2_input_acceleration
   + 0.5377163887023926 x3_input_acceleration >= -0.1396052743617432
 CBF_Constraint_sphere_right: 0.285685658454895 x1_input_acceleration
   - 1.047747850418091 x2_input_acceleration
   + 0.3377165794372559 x3_input_acceleration >= -0.2927360853823343
 Relaxed_CLF_constraint: 0.0288431644439697 x1_input_acceleration
   - 0.2348737716674805 x2_input_acceleration
   + 0.0298581719398499 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.28631991147995 x1_input_acceleration
   - 0.1954762935638428 x2_input_acceleration
   + 0.3383431434631348 x3_input_acceleration >= -0.0280415389459649
 CBF_Constraint_sphere_low: 0.1863201260566711 x1_input_acceleration
   - 0.5954759120941162 x2_input_acceleration
   + 0.5383429527282715 x3_input_acceleration >= -0.1391549715025719
 CBF_Constraint_sphere_right: 0.2863200902938843 x1_input_acceleration
   - 1.045475482940674 x2_input_acceleration
   + 0.3383431434631348 x3_input_acceleration >= -0.2917435655661121
 Relaxed_CLF_constraint: 0.0291603803634644 x1_input_acceleration
   - 0.233737587928772 x2_input_acceleration
   + 0.0301714539527893 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2869542241096497 x1_input_acceleration
   - 0.1932041645050049 x2_input_acceleration
   + 0.3389694690704346 x3_input_acceleration >= -0.0280177192195851
 CBF_Constraint_sphere_low: 0.1869544386863708 x1_input_acceleration
   - 0.5932037830352783 x2_input_acceleration
   + 0.5389692783355713 x3_input_acceleration >= -0.138707643335367
 CBF_Constraint_sphere_right: 0.286954402923584 x1_input_acceleration
   - 1.043203353881836 x2_input_acceleration
   + 0.3389694690704346 x3_input_acceleration >= -0.2907540919673585
 Relaxed_CLF_constraint: 0.0294775366783142 x1_input_acceleration
   - 0.232601523399353 x2_input_acceleration
   + 0.0304846167564392 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2875036597251892 x1_input_acceleration
   - 0.1909544467926025 x2_input_acceleration
   + 0.3395109176635742 x3_input_acceleration >= -0.0279724045795021
 CBF_Constraint_sphere_low: 0.1875038743019104 x1_input_acceleration
   - 0.590954065322876 x2_input_acceleration
   + 0.5395107269287109 x3_input_acceleration >= -0.1382390586677517
 CBF_Constraint_sphere_right: 0.2875038385391235 x1_input_acceleration
   - 1.040953636169434 x2_input_acceleration
   + 0.3395109176635742 x3_input_acceleration >= -0.2897526482604645
 Relaxed_CLF_constraint: 0.029752254486084 x1_input_acceleration
   - 0.2314766645431519 x2_input_acceleration
   + 0.030755341053009 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2880530953407288 x1_input_acceleration
   - 0.1887047290802002 x2_input_acceleration
   + 0.3400524854660034 x3_input_acceleration >= -0.0279299383460533
 CBF_Constraint_sphere_low: 0.18805330991745 x1_input_acceleration
   - 0.5887043476104736 x2_input_acceleration
   + 0.5400522947311401 x3_input_acceleration >= -0.137773334327688
 CBF_Constraint_sphere_right: 0.2880532741546631 x1_input_acceleration
   - 1.038703918457031 x2_input_acceleration
   + 0.3400524854660034 x3_input_acceleration >= -0.2887540529602047
 Relaxed_CLF_constraint: 0.0300269722938538 x1_input_acceleration
   - 0.2303518056869507 x2_input_acceleration
   + 0.0310261249542236 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2886026501655579 x1_input_acceleration
   - 0.1864550113677979 x2_input_acceleration
   + 0.3405939340591431 x3_input_acceleration >= -0.0278902972161582
 CBF_Constraint_sphere_low: 0.1886028647422791 x1_input_acceleration
   - 0.5864546298980713 x2_input_acceleration
   + 0.5405937433242798 x3_input_acceleration >= -0.1373104172098089
 CBF_Constraint_sphere_right: 0.2886028289794922 x1_input_acceleration
   - 1.036454200744629 x2_input_acceleration
   + 0.3405939340591431 x3_input_acceleration >= -0.2877582827635093
 Relaxed_CLF_constraint: 0.0303017497062683 x1_input_acceleration
   - 0.2292269468307495 x2_input_acceleration
   + 0.0312968492507935 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2891520857810974 x1_input_acceleration
   - 0.1842052936553955 x2_input_acceleration
   + 0.3411355018615723 x3_input_acceleration >= -0.0278534873881441
 CBF_Constraint_sphere_low: 0.1891523003578186 x1_input_acceleration
   - 0.5842049121856689 x2_input_acceleration
   + 0.541135311126709 x3_input_acceleration >= -0.1368503492751799
 CBF_Constraint_sphere_right: 0.2891522645950317 x1_input_acceleration
   - 1.034204483032227 x2_input_acceleration
   + 0.3411355018615723 x3_input_acceleration >= -0.2867653438686842
 Relaxed_CLF_constraint: 0.0305764675140381 x1_input_acceleration
   - 0.2281020879745483 x2_input_acceleration
   + 0.0315676331520081 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2897016406059265 x1_input_acceleration
   - 0.1819555759429932 x2_input_acceleration
   + 0.3416769504547119 x3_input_acceleration >= -0.0278195026646359
 CBF_Constraint_sphere_low: 0.1897018551826477 x1_input_acceleration
   - 0.5819551944732666 x2_input_acceleration
   + 0.5416767597198486 x3_input_acceleration >= -0.1363930885636876
 CBF_Constraint_sphere_right: 0.2897018194198608 x1_input_acceleration
   - 1.031954765319824 x2_input_acceleration
   + 0.3416769504547119 x3_input_acceleration >= -0.2857752300783757
 Relaxed_CLF_constraint: 0.0308512449264526 x1_input_acceleration
   - 0.2269772291183472 x2_input_acceleration
   + 0.0318383574485779 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2901665568351746 x1_input_acceleration
   - 0.179729700088501 x2_input_acceleration
   + 0.3421337604522705 x3_input_acceleration >= -0.0277637264601737
 CBF_Constraint_sphere_low: 0.1901667714118958 x1_input_acceleration
   - 0.5797293186187744 x2_input_acceleration
   + 0.5421335697174072 x3_input_acceleration >= -0.1359145728074879
 CBF_Constraint_sphere_right: 0.2901667356491089 x1_input_acceleration
   - 1.029728889465332 x2_input_acceleration
   + 0.3421337604522705 x3_input_acceleration >= -0.2847734575794943
 Relaxed_CLF_constraint: 0.0310837030410767 x1_input_acceleration
   - 0.2258642911911011 x2_input_acceleration
   + 0.0320667624473572 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2906314730644226 x1_input_acceleration
   - 0.1775038242340088 x2_input_acceleration
   + 0.3425906896591187 x3_input_acceleration >= -0.0277106603486012
 CBF_Constraint_sphere_low: 0.1906316876411438 x1_input_acceleration
   - 0.5775034427642822 x2_input_acceleration
   + 0.5425904989242554 x3_input_acceleration >= -0.1354387790650954
 CBF_Constraint_sphere_right: 0.2906316518783569 x1_input_acceleration
   - 1.02750301361084 x2_input_acceleration
   + 0.3425906896591187 x3_input_acceleration >= -0.2837743951735027
 Relaxed_CLF_constraint: 0.0313161611557007 x1_input_acceleration
   - 0.224751353263855 x2_input_acceleration
   + 0.0322952270507812 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2910963892936707 x1_input_acceleration
   - 0.1752779483795166 x2_input_acceleration
   + 0.3430474996566772 x3_input_acceleration >= -0.0276602635171607
 CBF_Constraint_sphere_low: 0.1910966038703918 x1_input_acceleration
   - 0.57527756690979 x2_input_acceleration
   + 0.543047308921814 x3_input_acceleration >= -0.1349656426819174
 CBF_Constraint_sphere_right: 0.291096568107605 x1_input_acceleration
   - 1.025277137756348 x2_input_acceleration
   + 0.3430474996566772 x3_input_acceleration >= -0.282778002047643
 Relaxed_CLF_constraint: 0.0315486192703247 x1_input_acceleration
   - 0.2236384153366089 x2_input_acceleration
   + 0.0325236320495605 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2915613055229187 x1_input_acceleration
   - 0.1730520725250244 x2_input_acceleration
   + 0.3435043096542358 x3_input_acceleration >= -0.0276125563586171
 CBF_Constraint_sphere_low: 0.1915615200996399 x1_input_acceleration
   - 0.5730516910552979 x2_input_acceleration
   + 0.5435041189193726 x3_input_acceleration >= -0.1344951959716362
 CBF_Constraint_sphere_right: 0.291561484336853 x1_input_acceleration
   - 1.023051261901855 x2_input_acceleration
   + 0.3435043096542358 x3_input_acceleration >= -0.2817842985946802
 Relaxed_CLF_constraint: 0.0317810773849487 x1_input_acceleration
   - 0.2225254774093628 x2_input_acceleration
   + 0.0327520370483398 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2920262217521667 x1_input_acceleration
   - 0.1708261966705322 x2_input_acceleration
   + 0.343961238861084 x3_input_acceleration >= -0.0275675593746542
 CBF_Constraint_sphere_low: 0.1920264363288879 x1_input_acceleration
   - 0.5708258152008057 x2_input_acceleration
   + 0.5439610481262207 x3_input_acceleration >= -0.1340274713568534
 CBF_Constraint_sphere_right: 0.2920264005661011 x1_input_acceleration
   - 1.020825386047363 x2_input_acceleration
   + 0.343961238861084 x3_input_acceleration >= -0.2807933053162982
 Relaxed_CLF_constraint: 0.0320135354995728 x1_input_acceleration
   - 0.2214125394821167 x2_input_acceleration
   + 0.0329805016517639 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2924904227256775 x1_input_acceleration
   - 0.1686005592346191 x2_input_acceleration
   + 0.3444173336029053 x3_input_acceleration >= -0.0275250239116517
 CBF_Constraint_sphere_low: 0.1924906373023987 x1_input_acceleration
   - 0.5686001777648926 x2_input_acceleration
   + 0.544417142868042 x3_input_acceleration >= -0.1335622082629883
 CBF_Constraint_sphere_right: 0.2924906015396118 x1_input_acceleration
   - 1.01859974861145 x2_input_acceleration
   + 0.3444173336029053 x3_input_acceleration >= -0.2798048748866121
 Relaxed_CLF_constraint: 0.0322456359863281 x1_input_acceleration
   - 0.2202997207641602 x2_input_acceleration
   + 0.0332085490226746 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2928711771965027 x1_input_acceleration
   - 0.1664001941680908 x2_input_acceleration
   + 0.344789981842041 x3_input_acceleration >= -0.0274606706575696
 CBF_Constraint_sphere_low: 0.1928713917732239 x1_input_acceleration
   - 0.5663998126983643 x2_input_acceleration
   + 0.5447897911071777 x3_input_acceleration >= -0.1330760095209712
 CBF_Constraint_sphere_right: 0.292871356010437 x1_input_acceleration
   - 1.016399383544922 x2_input_acceleration
   + 0.344789981842041 x3_input_acceleration >= -0.2788053674051311
 Relaxed_CLF_constraint: 0.0324360132217407 x1_input_acceleration
   - 0.219199538230896 x2_input_acceleration
   + 0.0333948731422424 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2932519316673279 x1_input_acceleration
   - 0.1641998291015625 x2_input_acceleration
   + 0.3451626300811768 x3_input_acceleration >= -0.0273988801270392
 CBF_Constraint_sphere_low: 0.1932521462440491 x1_input_acceleration
   - 0.5641994476318359 x2_input_acceleration
   + 0.5451624393463135 x3_input_acceleration >= -0.1325923735025058
 CBF_Constraint_sphere_right: 0.2932521104812622 x1_input_acceleration
   - 1.014199018478394 x2_input_acceleration
   + 0.3451626300811768 x3_input_acceleration >= -0.2778084226472016
 Relaxed_CLF_constraint: 0.0326263904571533 x1_input_acceleration
   - 0.2180993556976318 x2_input_acceleration
   + 0.0335811972618103 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2936326861381531 x1_input_acceleration
   - 0.1619994640350342 x2_input_acceleration
   + 0.3455352783203125 x3_input_acceleration >= -0.0273396523200603
 CBF_Constraint_sphere_low: 0.1936329007148743 x1_input_acceleration
   - 0.5619990825653076 x2_input_acceleration
   + 0.5455350875854492 x3_input_acceleration >= -0.1321113002075919
 CBF_Constraint_sphere_right: 0.2936328649520874 x1_input_acceleration
   - 1.011998653411865 x2_input_acceleration
   + 0.3455352783203125 x3_input_acceleration >= -0.2768140406128238
 Relaxed_CLF_constraint: 0.0328167676925659 x1_input_acceleration
   - 0.2169991731643677 x2_input_acceleration
   + 0.0337675213813782 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2940134406089783 x1_input_acceleration
   - 0.1597990989685059 x2_input_acceleration
   + 0.3459079265594482 x3_input_acceleration >= -0.027282987236633
 CBF_Constraint_sphere_low: 0.1940136551856995 x1_input_acceleration
   - 0.5597987174987793 x2_input_acceleration
   + 0.545907735824585 x3_input_acceleration >= -0.1316327896362296
 CBF_Constraint_sphere_right: 0.2940136194229126 x1_input_acceleration
   - 1.009798288345337 x2_input_acceleration
   + 0.3459079265594482 x3_input_acceleration >= -0.2758222213019975
 Relaxed_CLF_constraint: 0.0330071449279785 x1_input_acceleration
   - 0.2158989906311035 x2_input_acceleration
   + 0.033953845500946 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2943941950798035 x1_input_acceleration
   - 0.1575987339019775 x2_input_acceleration
   + 0.346280574798584 x3_input_acceleration >= -0.0272288848767574
 CBF_Constraint_sphere_low: 0.1943944096565247 x1_input_acceleration
   - 0.557598352432251 x2_input_acceleration
   + 0.5462803840637207 x3_input_acceleration >= -0.1311568417884189
 CBF_Constraint_sphere_right: 0.2943943738937378 x1_input_acceleration
   - 1.007597923278809 x2_input_acceleration
   + 0.346280574798584 x3_input_acceleration >= -0.2748329647147228
 Relaxed_CLF_constraint: 0.0331975221633911 x1_input_acceleration
   - 0.2147988080978394 x2_input_acceleration
   + 0.0341401696205139 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2947484850883484 x1_input_acceleration
   - 0.1554069519042969 x2_input_acceleration
   + 0.3466266393661499 x3_input_acceleration >= -0.0271695043200829
 CBF_Constraint_sphere_low: 0.1947486996650696 x1_input_acceleration
   - 0.5554065704345703 x2_input_acceleration
   + 0.5466264486312866 x3_input_acceleration >= -0.1306759972115949
 CBF_Constraint_sphere_right: 0.2947486639022827 x1_input_acceleration
   - 1.005406141281128 x2_input_acceleration
   + 0.3466266393661499 x3_input_acceleration >= -0.2738420777290648
 Relaxed_CLF_constraint: 0.0333746671676636 x1_input_acceleration
   - 0.213702917098999 x2_input_acceleration
   + 0.0343132019042969 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2950454354286194 x1_input_acceleration
   - 0.153233528137207 x2_input_acceleration
   + 0.3469153642654419 x3_input_acceleration >= -0.027095648269016
 CBF_Constraint_sphere_low: 0.1950456500053406 x1_input_acceleration
   - 0.5532331466674805 x2_input_acceleration
   + 0.5469151735305786 x3_input_acceleration >= -0.1301814817988977
 CBF_Constraint_sphere_right: 0.2950456142425537 x1_input_acceleration
   - 1.003232717514038 x2_input_acceleration
   + 0.3469153642654419 x3_input_acceleration >= -0.2728445174844479
 Relaxed_CLF_constraint: 0.0335231423377991 x1_input_acceleration
   - 0.2126162052154541 x2_input_acceleration
   + 0.0344575643539429 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2953423857688904 x1_input_acceleration
   - 0.1510601043701172 x2_input_acceleration
   + 0.3472040891647339 x3_input_acceleration >= -0.0270242398741708
 CBF_Constraint_sphere_low: 0.1953426003456116 x1_input_acceleration
   - 0.5510597229003906 x2_input_acceleration
   + 0.5472038984298706 x3_input_acceleration >= -0.1296894140424222
 CBF_Constraint_sphere_right: 0.2953425645828247 x1_input_acceleration
   - 1.001059293746948 x2_input_acceleration
   + 0.3472040891647339 x3_input_acceleration >= -0.2718494048960528
 Relaxed_CLF_constraint: 0.0336716175079346 x1_input_acceleration
   - 0.2115294933319092 x2_input_acceleration
   + 0.0346019268035889 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2956393361091614 x1_input_acceleration
   - 0.1488866806030273 x2_input_acceleration
   + 0.3474928140640259 x3_input_acceleration >= -0.0269552791355473
 CBF_Constraint_sphere_low: 0.1956395506858826 x1_input_acceleration
   - 0.5488862991333008 x2_input_acceleration
   + 0.5474926233291626 x3_input_acceleration >= -0.1291997939421684
 CBF_Constraint_sphere_right: 0.2956395149230957 x1_input_acceleration
   - 0.9988858699798584 x2_input_acceleration
   + 0.3474928140640259 x3_input_acceleration >= -0.2708567399638794
 Relaxed_CLF_constraint: 0.0338200926780701 x1_input_acceleration
   - 0.2104427814483643 x2_input_acceleration
   + 0.0347462892532349 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2959362864494324 x1_input_acceleration
   - 0.1467132568359375 x2_input_acceleration
   + 0.3477815389633179 x3_input_acceleration >= -0.0268887660531455
 CBF_Constraint_sphere_low: 0.1959365010261536 x1_input_acceleration
   - 0.5467128753662109 x2_input_acceleration
   + 0.5477813482284546 x3_input_acceleration >= -0.1287126214981364
 CBF_Constraint_sphere_right: 0.2959364652633667 x1_input_acceleration
   - 0.9967124462127686 x2_input_acceleration
   + 0.3477815389633179 x3_input_acceleration >= -0.2698665226879277
 Relaxed_CLF_constraint: 0.0339685678482056 x1_input_acceleration
   - 0.2093560695648193 x2_input_acceleration
   + 0.0348906517028809 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2962332367897034 x1_input_acceleration
   - 0.1445398330688477 x2_input_acceleration
   + 0.3480703830718994 x3_input_acceleration >= -0.0268247213735734
 CBF_Constraint_sphere_low: 0.1962334513664246 x1_input_acceleration
   - 0.5445394515991211 x2_input_acceleration
   + 0.5480701923370361 x3_input_acceleration >= -0.1282279293778516
 CBF_Constraint_sphere_right: 0.2962334156036377 x1_input_acceleration
   - 0.9945390224456787 x2_input_acceleration
   + 0.3480703830718994 x3_input_acceleration >= -0.2688787738148056
 Relaxed_CLF_constraint: 0.0341170430183411 x1_input_acceleration
   - 0.2082693576812744 x2_input_acceleration
   + 0.0350350737571716 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2965301871299744 x1_input_acceleration
   - 0.1423664093017578 x2_input_acceleration
   + 0.3483591079711914 x3_input_acceleration >= -0.0267631036208244
 CBF_Constraint_sphere_low: 0.1965304017066956 x1_input_acceleration
   - 0.5423660278320312 x2_input_acceleration
   + 0.5483589172363281 x3_input_acceleration >= -0.1277456522634723
 CBF_Constraint_sphere_right: 0.2965303659439087 x1_input_acceleration
   - 0.9923655986785889 x2_input_acceleration
   + 0.3483591079711914 x3_input_acceleration >= -0.2678934518685067
 Relaxed_CLF_constraint: 0.0342655181884766 x1_input_acceleration
   - 0.2071826457977295 x2_input_acceleration
   + 0.0351794362068176 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2967438101768494 x1_input_acceleration
   - 0.1402215957641602 x2_input_acceleration
   + 0.3485643863677979 x3_input_acceleration >= -0.0266790290602203
 CBF_Constraint_sphere_low: 0.1967440247535706 x1_input_acceleration
   - 0.5402212142944336 x2_input_acceleration
   + 0.5485641956329346 x3_input_acceleration >= -0.1272424620950986
 CBF_Constraint_sphere_right: 0.2967439889907837 x1_input_acceleration
   - 0.9902207851409912 x2_input_acceleration
   + 0.3485643863677979 x3_input_acceleration >= -0.2668978324428407
 Relaxed_CLF_constraint: 0.0343723297119141 x1_input_acceleration
   - 0.2061102390289307 x2_input_acceleration
   + 0.0352820754051208 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2969573140144348 x1_input_acceleration
   - 0.1380767822265625 x2_input_acceleration
   + 0.3487697839736938 x3_input_acceleration >= -0.0265973015874416
 CBF_Constraint_sphere_low: 0.196957528591156 x1_input_acceleration
   - 0.5380764007568359 x2_input_acceleration
   + 0.5487695932388306 x3_input_acceleration >= -0.1267416368959197
 CBF_Constraint_sphere_right: 0.2969574928283691 x1_input_acceleration
   - 0.9880759716033936 x2_input_acceleration
   + 0.3487697839736938 x3_input_acceleration >= -0.2659045601049895
 Relaxed_CLF_constraint: 0.0344790816307068 x1_input_acceleration
   - 0.2050378322601318 x2_input_acceleration
   + 0.0353847742080688 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2971709370613098 x1_input_acceleration
   - 0.1359319686889648 x2_input_acceleration
   + 0.3489750623703003 x3_input_acceleration >= -0.0265179150254774
 CBF_Constraint_sphere_low: 0.197171151638031 x1_input_acceleration
   - 0.5359315872192383 x2_input_acceleration
   + 0.548974871635437 x3_input_acceleration >= -0.126243134726186
 CBF_Constraint_sphere_right: 0.2971711158752441 x1_input_acceleration
   - 0.9859311580657959 x2_input_acceleration
   + 0.3489750623703003 x3_input_acceleration >= -0.2649136286779635
 Relaxed_CLF_constraint: 0.0345858931541443 x1_input_acceleration
   - 0.203965425491333 x2_input_acceleration
   + 0.0354874134063721 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2973845601081848 x1_input_acceleration
   - 0.1337871551513672 x2_input_acceleration
   + 0.3491803407669067 x3_input_acceleration >= -0.0264408724630819
 CBF_Constraint_sphere_low: 0.197384774684906 x1_input_acceleration
   - 0.5337867736816406 x2_input_acceleration
   + 0.5491801500320435 x3_input_acceleration >= -0.125746976556021
 CBF_Constraint_sphere_right: 0.2973847389221191 x1_input_acceleration
   - 0.9837863445281982 x2_input_acceleration
   + 0.3491803407669067 x3_input_acceleration >= -0.2639250412505062
 Relaxed_CLF_constraint: 0.0346927046775818 x1_input_acceleration
   - 0.2028930187225342 x2_input_acceleration
   + 0.0355900526046753 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2975981831550598 x1_input_acceleration
   - 0.1316423416137695 x2_input_acceleration
   + 0.3493856191635132 x3_input_acceleration >= -0.0263661739002551
 CBF_Constraint_sphere_low: 0.197598397731781 x1_input_acceleration
   - 0.531641960144043 x2_input_acceleration
   + 0.5493854284286499 x3_input_acceleration >= -0.1252531623854246
 CBF_Constraint_sphere_right: 0.2975983619689941 x1_input_acceleration
   - 0.9816415309906006 x2_input_acceleration
   + 0.3493856191635132 x3_input_acceleration >= -0.2629387978226175
 Relaxed_CLF_constraint: 0.0347995162010193 x1_input_acceleration
   - 0.2018206119537354 x2_input_acceleration
   + 0.0356926918029785 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2978116869926453 x1_input_acceleration
   - 0.1294975280761719 x2_input_acceleration
   + 0.3495908975601196 x3_input_acceleration >= -0.0262938015860336
 CBF_Constraint_sphere_low: 0.1978119015693665 x1_input_acceleration
   - 0.5294971466064453 x2_input_acceleration
   + 0.5495907068252563 x3_input_acceleration >= -0.1247616804238853
 CBF_Constraint_sphere_right: 0.2978118658065796 x1_input_acceleration
   - 0.9794967174530029 x2_input_acceleration
   + 0.3495908975601196 x3_input_acceleration >= -0.2619548806433235
 Relaxed_CLF_constraint: 0.034906268119812 x1_input_acceleration
   - 0.2007482051849365 x2_input_acceleration
   + 0.0357953310012817 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2980251908302307 x1_input_acceleration
   - 0.1273527145385742 x2_input_acceleration
   + 0.3497960567474365 x3_input_acceleration >= -0.0262237523964487
 CBF_Constraint_sphere_low: 0.1980254054069519 x1_input_acceleration
   - 0.5273523330688477 x2_input_acceleration
   + 0.5497958660125732 x3_input_acceleration >= -0.124272509666065
 CBF_Constraint_sphere_right: 0.298025369644165 x1_input_acceleration
   - 0.9773519039154053 x2_input_acceleration
   + 0.3497960567474365 x3_input_acceleration >= -0.2609732865886661
 Relaxed_CLF_constraint: 0.0350130200386047 x1_input_acceleration
   - 0.1996757984161377 x2_input_acceleration
   + 0.0358979105949402 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2981557250022888 x1_input_acceleration
   - 0.1252377033233643 x2_input_acceleration
   + 0.3499182462692261 x3_input_acceleration >= -0.0261310244393706
 CBF_Constraint_sphere_low: 0.19815593957901 x1_input_acceleration
   - 0.5252373218536377 x2_input_acceleration
   + 0.5499180555343628 x3_input_acceleration >= -0.1237624721152791
 CBF_Constraint_sphere_right: 0.2981559038162231 x1_input_acceleration
   - 0.9752368927001953 x2_input_acceleration
   + 0.3499182462692261 x3_input_acceleration >= -0.259981679734033
 Relaxed_CLF_constraint: 0.0350782871246338 x1_input_acceleration
   - 0.1986182928085327 x2_input_acceleration
   + 0.035959005355835 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2982862591743469 x1_input_acceleration
   - 0.1231226921081543 x2_input_acceleration
   + 0.3500404357910156 x3_input_acceleration >= -0.0260405491032373
 CBF_Constraint_sphere_low: 0.1982864737510681 x1_input_acceleration
   - 0.5231223106384277 x2_input_acceleration
   + 0.5500402450561523 x3_input_acceleration >= -0.1232546871854382
 CBF_Constraint_sphere_right: 0.2982864379882812 x1_input_acceleration
   - 0.9731218814849854 x2_input_acceleration
   + 0.3500404357910156 x3_input_acceleration >= -0.2589923255003447
 Relaxed_CLF_constraint: 0.0351435542106628 x1_input_acceleration
   - 0.1975607872009277 x2_input_acceleration
   + 0.0360201001167297 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.298416793346405 x1_input_acceleration
   - 0.1210076808929443 x2_input_acceleration
   + 0.3501627445220947 x3_input_acceleration >= -0.0259523472593714
 CBF_Constraint_sphere_low: 0.1984170079231262 x1_input_acceleration
   - 0.5210072994232178 x2_input_acceleration
   + 0.5501625537872314 x3_input_acceleration >= -0.1227491876687822
 CBF_Constraint_sphere_right: 0.2984169721603394 x1_input_acceleration
   - 0.9710068702697754 x2_input_acceleration
   + 0.3501627445220947 x3_input_acceleration >= -0.2580052447589237
 Relaxed_CLF_constraint: 0.0352088212966919 x1_input_acceleration
   - 0.1965032815933228 x2_input_acceleration
   + 0.0360812544822693 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2985473275184631 x1_input_acceleration
   - 0.1188926696777344 x2_input_acceleration
   + 0.3502849340438843 x3_input_acceleration >= -0.0258663771724109
 CBF_Constraint_sphere_low: 0.1985475420951843 x1_input_acceleration
   - 0.5188922882080078 x2_input_acceleration
   + 0.550284743309021 x3_input_acceleration >= -0.1222459079881141
 CBF_Constraint_sphere_right: 0.2985475063323975 x1_input_acceleration
   - 0.9688918590545654 x2_input_acceleration
   + 0.3502849340438843 x3_input_acceleration >= -0.2570203957744083
 Relaxed_CLF_constraint: 0.0352740883827209 x1_input_acceleration
   - 0.1954457759857178 x2_input_acceleration
   + 0.0361423492431641 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2986778616905212 x1_input_acceleration
   - 0.1167776584625244 x2_input_acceleration
   + 0.3504071235656738 x3_input_acceleration >= -0.0257826597063954
 CBF_Constraint_sphere_low: 0.1986780762672424 x1_input_acceleration
   - 0.5167772769927979 x2_input_acceleration
   + 0.5504069328308105 x3_input_acceleration >= -0.1217448809283909
 CBF_Constraint_sphere_right: 0.2986780405044556 x1_input_acceleration
   - 0.9667768478393555 x2_input_acceleration
   + 0.3504071235656738 x3_input_acceleration >= -0.2560377994108377
 Relaxed_CLF_constraint: 0.03533935546875 x1_input_acceleration
   - 0.1943882703781128 x2_input_acceleration
   + 0.0362034440040588 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2988083958625793 x1_input_acceleration
   - 0.1146626472473145 x2_input_acceleration
   + 0.3505293130874634 x3_input_acceleration >= -0.0257011948613247
 CBF_Constraint_sphere_low: 0.1988086104393005 x1_input_acceleration
   - 0.5146622657775879 x2_input_acceleration
   + 0.5505291223526001 x3_input_acceleration >= -0.1212461064896126
 CBF_Constraint_sphere_right: 0.2988085746765137 x1_input_acceleration
   - 0.9646618366241455 x2_input_acceleration
   + 0.3505293130874634 x3_input_acceleration >= -0.255057455668212
 Relaxed_CLF_constraint: 0.0354046225547791 x1_input_acceleration
   - 0.1933307647705078 x2_input_acceleration
   + 0.0362645387649536 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2989389300346375 x1_input_acceleration
   - 0.1125476360321045 x2_input_acceleration
   + 0.3506515026092529 x3_input_acceleration >= -0.025621982637199
 CBF_Constraint_sphere_low: 0.1989391446113586 x1_input_acceleration
   - 0.5125472545623779 x2_input_acceleration
   + 0.5506513118743896 x3_input_acceleration >= -0.1207495846717791
 CBF_Constraint_sphere_right: 0.2989391088485718 x1_input_acceleration
   - 0.9625468254089355 x2_input_acceleration
   + 0.3506515026092529 x3_input_acceleration >= -0.2540793645465312
 Relaxed_CLF_constraint: 0.0354698896408081 x1_input_acceleration
   - 0.1922732591629028 x2_input_acceleration
   + 0.0363256335258484 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2990314364433289 x1_input_acceleration
   - 0.1104469299316406 x2_input_acceleration
   + 0.3507357835769653 x3_input_acceleration >= -0.0255334785485083
 CBF_Constraint_sphere_low: 0.19903165102005 x1_input_acceleration
   - 0.5104465484619141 x2_input_acceleration
   + 0.5507355928421021 x3_input_acceleration >= -0.1202447425418974
 CBF_Constraint_sphere_right: 0.2990316152572632 x1_input_acceleration
   - 0.9604461193084717 x2_input_acceleration
   + 0.3507357835769653 x3_input_acceleration >= -0.2530980612248547
 Relaxed_CLF_constraint: 0.0355161428451538 x1_input_acceleration
   - 0.1912229061126709 x2_input_acceleration
   + 0.0363677740097046 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2990793585777283 x1_input_acceleration
   - 0.1083633899688721 x2_input_acceleration
   + 0.3507753610610962 x3_input_acceleration >= -0.0254336102350884
 CBF_Constraint_sphere_low: 0.1990795731544495 x1_input_acceleration
   - 0.5083630084991455 x2_input_acceleration
   + 0.5507751703262329 x3_input_acceleration >= -0.1197297282763877
 CBF_Constraint_sphere_right: 0.2990795373916626 x1_input_acceleration
   - 0.9583625793457031 x2_input_acceleration
   + 0.3507753610610962 x3_input_acceleration >= -0.2521126892760256
 Relaxed_CLF_constraint: 0.0355401039123535 x1_input_acceleration
   - 0.1901811361312866 x2_input_acceleration
   + 0.03638756275177 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2991272807121277 x1_input_acceleration
   - 0.1062796115875244 x2_input_acceleration
   + 0.3508149385452271 x3_input_acceleration >= -0.0253359017529796
 CBF_Constraint_sphere_low: 0.1991274952888489 x1_input_acceleration
   - 0.5062792301177979 x2_input_acceleration
   + 0.5508147478103638 x3_input_acceleration >= -0.1192168261585188
 CBF_Constraint_sphere_right: 0.299127459526062 x1_input_acceleration
   - 0.9562788009643555 x2_input_acceleration
   + 0.3508149385452271 x3_input_acceleration >= -0.2511293758307082
 Relaxed_CLF_constraint: 0.0355640649795532 x1_input_acceleration
   - 0.1891392469406128 x2_input_acceleration
   + 0.0364073514938354 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2991752028465271 x1_input_acceleration
   - 0.1041960716247559 x2_input_acceleration
   + 0.3508545160293579 x3_input_acceleration >= -0.0252403786896217
 CBF_Constraint_sphere_low: 0.1991754174232483 x1_input_acceleration
   - 0.5041956901550293 x2_input_acceleration
   + 0.5508543252944946 x3_input_acceleration >= -0.1187061571430711
 CBF_Constraint_sphere_right: 0.2991753816604614 x1_input_acceleration
   - 0.9541952610015869 x2_input_acceleration
   + 0.3508545160293579 x3_input_acceleration >= -0.2501483491319411
 Relaxed_CLF_constraint: 0.0355880260467529 x1_input_acceleration
   - 0.1880974769592285 x2_input_acceleration
   + 0.0364271402359009 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.299223005771637 x1_input_acceleration
   - 0.1021125316619873 x2_input_acceleration
   + 0.3508939743041992 x3_input_acceleration >= -0.0251469893771074
 CBF_Constraint_sphere_low: 0.1992232203483582 x1_input_acceleration
   - 0.5021121501922607 x2_input_acceleration
   + 0.5508937835693359 x3_input_acceleration >= -0.1181976159180011
 CBF_Constraint_sphere_right: 0.2992231845855713 x1_input_acceleration
   - 0.9521117210388184 x2_input_acceleration
   + 0.3508939743041992 x3_input_acceleration >= -0.2491694561840069
 Relaxed_CLF_constraint: 0.0356119275093079 x1_input_acceleration
   - 0.1870557069778442 x2_input_acceleration
   + 0.0364468693733215 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2992709279060364 x1_input_acceleration
   - 0.1000289916992188 x2_input_acceleration
   + 0.3509334325790405 x3_input_acceleration >= -0.0250557903929526
 CBF_Constraint_sphere_low: 0.1992711424827576 x1_input_acceleration
   - 0.5000286102294922 x2_input_acceleration
   + 0.5509332418441772 x3_input_acceleration >= -0.1176912590608389
 CBF_Constraint_sphere_right: 0.2992711067199707 x1_input_acceleration
   - 0.9500281810760498 x2_input_acceleration
   + 0.3509334325790405 x3_input_acceleration >= -0.248192753564443
 Relaxed_CLF_constraint: 0.0356358885765076 x1_input_acceleration
   - 0.18601393699646 x2_input_acceleration
   + 0.0364665985107422 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2993188500404358 x1_input_acceleration
   - 0.0979454517364502 x2_input_acceleration
   + 0.3509730100631714 x3_input_acceleration >= -0.0249667848245473
 CBF_Constraint_sphere_low: 0.199319064617157 x1_input_acceleration
   - 0.4979450702667236 x2_input_acceleration
   + 0.5509728193283081 x3_input_acceleration >= -0.1171871075403438
 CBF_Constraint_sphere_right: 0.2993190288543701 x1_input_acceleration
   - 0.9479446411132812 x2_input_acceleration
   + 0.3509730100631714 x3_input_acceleration >= -0.2472182443606285
 Relaxed_CLF_constraint: 0.0356598496437073 x1_input_acceleration
   - 0.1849721670150757 x2_input_acceleration
   + 0.0364863872528076 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2993667721748352 x1_input_acceleration
   - 0.0958619117736816 x2_input_acceleration
   + 0.3510125875473022 x3_input_acceleration >= -0.0248799517569843
 CBF_Constraint_sphere_low: 0.1993669867515564 x1_input_acceleration
   - 0.4958615303039551 x2_input_acceleration
   + 0.551012396812439 x3_input_acceleration >= -0.116685128520691
 CBF_Constraint_sphere_right: 0.2993669509887695 x1_input_acceleration
   - 0.9458611011505127 x2_input_acceleration
   + 0.3510125875473022 x3_input_acceleration >= -0.2462459076576563
 Relaxed_CLF_constraint: 0.035683810710907 x1_input_acceleration
   - 0.1839303970336914 x2_input_acceleration
   + 0.036506175994873 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2994145750999451 x1_input_acceleration
   - 0.0937783718109131 x2_input_acceleration
   + 0.3510520458221436 x3_input_acceleration >= -0.0247952524194247
 CBF_Constraint_sphere_low: 0.1994147896766663 x1_input_acceleration
   - 0.4937779903411865 x2_input_acceleration
   + 0.5510518550872803 x3_input_acceleration >= -0.1161852772705757
 CBF_Constraint_sphere_right: 0.2994147539138794 x1_input_acceleration
   - 0.9437775611877441 x2_input_acceleration
   + 0.3510520458221436 x3_input_acceleration >= -0.2452757046846769
 Relaxed_CLF_constraint: 0.0357077121734619 x1_input_acceleration
   - 0.1828886270523071 x2_input_acceleration
   + 0.0365259051322937 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2993816733360291 x1_input_acceleration
   - 0.0917270183563232 x2_input_acceleration
   + 0.351010799407959 x3_input_acceleration >= -0.0246879533817591
 CBF_Constraint_sphere_low: 0.1993818879127502 x1_input_acceleration
   - 0.4917266368865967 x2_input_acceleration
   + 0.5510106086730957 x3_input_acceleration >= -0.1156652283804377
 CBF_Constraint_sphere_right: 0.2993818521499634 x1_input_acceleration
   - 0.9417262077331543 x2_input_acceleration
   + 0.351010799407959 x3_input_acceleration >= -0.2442965812573063
 Relaxed_CLF_constraint: 0.0356912612915039 x1_input_acceleration
   - 0.1818629503250122 x2_input_acceleration
   + 0.0365052819252014 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2993473410606384 x1_input_acceleration
   - 0.0896761417388916 x2_input_acceleration
   + 0.3509680032730103 x3_input_acceleration >= -0.0245822950796729
 CBF_Constraint_sphere_low: 0.1993475556373596 x1_input_acceleration
   - 0.489675760269165 x2_input_acceleration
   + 0.550967812538147 x3_input_acceleration >= -0.1151468321467114
 CBF_Constraint_sphere_right: 0.2993475198745728 x1_input_acceleration
   - 0.9396753311157227 x2_input_acceleration
   + 0.3509680032730103 x3_input_acceleration >= -0.2433193012209861
 Relaxed_CLF_constraint: 0.0356740951538086 x1_input_acceleration
   - 0.1808375120162964 x2_input_acceleration
   + 0.0364838838577271 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2993128895759583 x1_input_acceleration
   - 0.08762526512146 x2_input_acceleration
   + 0.3509252071380615 x3_input_acceleration >= -0.0244787234897018
 CBF_Constraint_sphere_low: 0.1993131041526794 x1_input_acceleration
   - 0.4876248836517334 x2_input_acceleration
   + 0.5509250164031982 x3_input_acceleration >= -0.1146305285855519
 CBF_Constraint_sphere_right: 0.2993130683898926 x1_input_acceleration
   - 0.937624454498291 x2_input_acceleration
   + 0.3509252071380615 x3_input_acceleration >= -0.2423441078967704
 Relaxed_CLF_constraint: 0.0356568694114685 x1_input_acceleration
   - 0.1798120737075806 x2_input_acceleration
   + 0.0364624857902527 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2992785573005676 x1_input_acceleration
   - 0.0855743885040283 x2_input_acceleration
   + 0.3508824110031128 x3_input_acceleration >= -0.0243772742947762
 CBF_Constraint_sphere_low: 0.1992787718772888 x1_input_acceleration
   - 0.4855740070343018 x2_input_acceleration
   + 0.5508822202682495 x3_input_acceleration >= -0.1141163414589862
 CBF_Constraint_sphere_right: 0.299278736114502 x1_input_acceleration
   - 0.9355735778808594 x2_input_acceleration
   + 0.3508824110031128 x3_input_acceleration >= -0.2413710369676109
 Relaxed_CLF_constraint: 0.0356397032737732 x1_input_acceleration
   - 0.1787866353988647 x2_input_acceleration
   + 0.0364410877227783 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.299244225025177 x1_input_acceleration
   - 0.0835235118865967 x2_input_acceleration
   + 0.3508396148681641 x3_input_acceleration >= -0.0242779296524077
 CBF_Constraint_sphere_low: 0.1992444396018982 x1_input_acceleration
   - 0.4835231304168701 x2_input_acceleration
   + 0.5508394241333008 x3_input_acceleration >= -0.1136042588849776
 CBF_Constraint_sphere_right: 0.2992444038391113 x1_input_acceleration
   - 0.9335227012634277 x2_input_acceleration
   + 0.3508396148681641 x3_input_acceleration >= -0.2404000705910085
 Relaxed_CLF_constraint: 0.0356225371360779 x1_input_acceleration
   - 0.1777611970901489 x2_input_acceleration
   + 0.036419689655304 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2992098927497864 x1_input_acceleration
   - 0.081472635269165 x2_input_acceleration
   + 0.3507968187332153 x3_input_acceleration >= -0.0241806895625964
 CBF_Constraint_sphere_low: 0.1992101073265076 x1_input_acceleration
   - 0.4814722537994385 x2_input_acceleration
   + 0.5507966279983521 x3_input_acceleration >= -0.1130942808635261
 CBF_Constraint_sphere_right: 0.2992100715637207 x1_input_acceleration
   - 0.9314718246459961 x2_input_acceleration
   + 0.3507968187332153 x3_input_acceleration >= -0.2394312087669631
 Relaxed_CLF_constraint: 0.0356053709983826 x1_input_acceleration
   - 0.1767357587814331 x2_input_acceleration
   + 0.0363982915878296 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2991754412651062 x1_input_acceleration
   - 0.0794217586517334 x2_input_acceleration
   + 0.3507540225982666 x3_input_acceleration >= -0.0240855361930926
 CBF_Constraint_sphere_low: 0.1991756558418274 x1_input_acceleration
   - 0.4794213771820068 x2_input_acceleration
   + 0.5507538318634033 x3_input_acceleration >= -0.112586395522834
 CBF_Constraint_sphere_right: 0.2991756200790405 x1_input_acceleration
   - 0.9294209480285645 x2_input_acceleration
   + 0.3507540225982666 x3_input_acceleration >= -0.2384644336632148
 Relaxed_CLF_constraint: 0.0355881452560425 x1_input_acceleration
   - 0.1757103204727173 x2_input_acceleration
   + 0.0363768935203552 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2991406321525574 x1_input_acceleration
   - 0.0773711204528809 x2_input_acceleration
   + 0.3507108688354492 x3_input_acceleration >= -0.0239923804010204
 CBF_Constraint_sphere_low: 0.1991408467292786 x1_input_acceleration
   - 0.4773707389831543 x2_input_acceleration
   + 0.5507106781005859 x3_input_acceleration >= -0.112080537561846
 CBF_Constraint_sphere_right: 0.2991408109664917 x1_input_acceleration
   - 0.9273703098297119 x2_input_acceleration
   + 0.3507108688354492 x3_input_acceleration >= -0.2374997574646655
 Relaxed_CLF_constraint: 0.0355707406997681 x1_input_acceleration
   - 0.174685001373291 x2_input_acceleration
   + 0.0363553166389465 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2991062998771667 x1_input_acceleration
   - 0.0753202438354492 x2_input_acceleration
   + 0.3506680727005005 x3_input_acceleration >= -0.0239014537422812
 CBF_Constraint_sphere_low: 0.1991065144538879 x1_input_acceleration
   - 0.4753198623657227 x2_input_acceleration
   + 0.5506678819656372 x3_input_acceleration >= -0.1115768729714667
 CBF_Constraint_sphere_right: 0.2991064786911011 x1_input_acceleration
   - 0.9253194332122803 x2_input_acceleration
   + 0.3506680727005005 x3_input_acceleration >= -0.2365372090716923
 Relaxed_CLF_constraint: 0.0355535745620728 x1_input_acceleration
   - 0.1736595630645752 x2_input_acceleration
   + 0.0363339185714722 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2990542054176331 x1_input_acceleration
   - 0.0732769966125488 x2_input_acceleration
   + 0.3506075143814087 x3_input_acceleration >= -0.0238071412878093
 CBF_Constraint_sphere_low: 0.1990544199943542 x1_input_acceleration
   - 0.4732766151428223 x2_input_acceleration
   + 0.5506073236465454 x3_input_acceleration >= -0.1110704603533869
 CBF_Constraint_sphere_right: 0.2990543842315674 x1_input_acceleration
   - 0.9232761859893799 x2_input_acceleration
   + 0.3506075143814087 x3_input_acceleration >= -0.2355745173709819
 Relaxed_CLF_constraint: 0.0355275273323059 x1_input_acceleration
   - 0.172637939453125 x2_input_acceleration
   + 0.0363036394119263 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.298937976360321 x1_input_acceleration
   - 0.0712604522705078 x2_input_acceleration
   + 0.3504828214645386 x3_input_acceleration >= -0.0236950434774862
 CBF_Constraint_sphere_low: 0.1989381909370422 x1_input_acceleration
   - 0.4712600708007812 x2_input_acceleration
   + 0.5504826307296753 x3_input_acceleration >= -0.1105483962198811
 CBF_Constraint_sphere_right: 0.2989381551742554 x1_input_acceleration
   - 0.9212596416473389 x2_input_acceleration
   + 0.3504828214645386 x3_input_acceleration >= -0.2346053890222285
 Relaxed_CLF_constraint: 0.0354694128036499 x1_input_acceleration
   - 0.1716296672821045 x2_input_acceleration
   + 0.0362412929534912 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.298821747303009 x1_input_acceleration
   - 0.0692439079284668 x2_input_acceleration
   + 0.3503581285476685 x3_input_acceleration >= -0.0235849934214635
 CBF_Constraint_sphere_low: 0.1988219618797302 x1_input_acceleration
   - 0.4692435264587402 x2_input_acceleration
   + 0.5503579378128052 x3_input_acceleration >= -0.1100283798406757
 CBF_Constraint_sphere_right: 0.2988219261169434 x1_input_acceleration
   - 0.9192430973052979 x2_input_acceleration
   + 0.3503581285476685 x3_input_acceleration >= -0.2336383084277755
 Relaxed_CLF_constraint: 0.0354112982749939 x1_input_acceleration
   - 0.170621395111084 x2_input_acceleration
   + 0.0361789464950562 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2987053990364075 x1_input_acceleration
   - 0.0672273635864258 x2_input_acceleration
   + 0.3502334356307983 x3_input_acceleration >= -0.0234769733155084
 CBF_Constraint_sphere_low: 0.1987056136131287 x1_input_acceleration
   - 0.4672269821166992 x2_input_acceleration
   + 0.5502332448959351 x3_input_acceleration >= -0.1095103993719896
 CBF_Constraint_sphere_right: 0.2987055778503418 x1_input_acceleration
   - 0.9172265529632568 x2_input_acceleration
   + 0.3502334356307983 x3_input_acceleration >= -0.2326732577833794
 Relaxed_CLF_constraint: 0.0353531241416931 x1_input_acceleration
   - 0.1696131229400635 x2_input_acceleration
   + 0.0361166 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2985891699790955 x1_input_acceleration
   - 0.0652108192443848 x2_input_acceleration
   + 0.3501087427139282 x3_input_acceleration >= -0.0233710187750141
 CBF_Constraint_sphere_low: 0.1985893845558167 x1_input_acceleration
   - 0.4652104377746582 x2_input_acceleration
   + 0.5501085519790649 x3_input_acceleration >= -0.1089944785083127
 CBF_Constraint_sphere_right: 0.2985893487930298 x1_input_acceleration
   - 0.9152100086212158 x2_input_acceleration
   + 0.3501087427139282 x3_input_acceleration >= -0.2317102727044548
 Relaxed_CLF_constraint: 0.0352950096130371 x1_input_acceleration
   - 0.168604850769043 x2_input_acceleration
   + 0.036054253578186 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2984729409217834 x1_input_acceleration
   - 0.0631942749023438 x2_input_acceleration
   + 0.3499840497970581 x3_input_acceleration >= -0.0232671119888203
 CBF_Constraint_sphere_low: 0.1984731554985046 x1_input_acceleration
   - 0.4631938934326172 x2_input_acceleration
   + 0.5499838590621948 x3_input_acceleration >= -0.1084806053989361
 CBF_Constraint_sphere_right: 0.2984731197357178 x1_input_acceleration
   - 0.9131934642791748 x2_input_acceleration
   + 0.3499840497970581 x3_input_acceleration >= -0.2307493353798306
 Relaxed_CLF_constraint: 0.0352368950843811 x1_input_acceleration
   - 0.1675965785980225 x2_input_acceleration
   + 0.035991907119751 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2983567118644714 x1_input_acceleration
   - 0.0611777305603027 x2_input_acceleration
   + 0.349859356880188 x3_input_acceleration >= -0.0231652529569267
 CBF_Constraint_sphere_low: 0.1983569264411926 x1_input_acceleration
   - 0.4611773490905762 x2_input_acceleration
   + 0.5498591661453247 x3_input_acceleration >= -0.1079687800438599
 CBF_Constraint_sphere_right: 0.2983568906784058 x1_input_acceleration
   - 0.9111769199371338 x2_input_acceleration
   + 0.349859356880188 x3_input_acceleration >= -0.2297904458095067
 Relaxed_CLF_constraint: 0.0351787805557251 x1_input_acceleration
   - 0.166588306427002 x2_input_acceleration
   + 0.0359295606613159 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2982403635978699 x1_input_acceleration
   - 0.0591611862182617 x2_input_acceleration
   + 0.3497346639633179 x3_input_acceleration >= -0.0230654239028191
 CBF_Constraint_sphere_low: 0.1982405781745911 x1_input_acceleration
   - 0.4591608047485352 x2_input_acceleration
   + 0.5497344732284546 x3_input_acceleration >= -0.1074589906270213
 CBF_Constraint_sphere_right: 0.2982405424118042 x1_input_acceleration
   - 0.9091603755950928 x2_input_acceleration
   + 0.3497346639633179 x3_input_acceleration >= -0.2288335862169581
 Relaxed_CLF_constraint: 0.0351206064224243 x1_input_acceleration
   - 0.1655800342559814 x2_input_acceleration
   + 0.0358672142028809 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2981241345405579 x1_input_acceleration
   - 0.0571446418762207 x2_input_acceleration
   + 0.3496099710464478 x3_input_acceleration >= -0.0229676603864541
 CBF_Constraint_sphere_low: 0.1981243491172791 x1_input_acceleration
   - 0.4571442604064941 x2_input_acceleration
   + 0.5496097803115845 x3_input_acceleration >= -0.1069512607874735
 CBF_Constraint_sphere_right: 0.2981243133544922 x1_input_acceleration
   - 0.9071438312530518 x2_input_acceleration
   + 0.3496099710464478 x3_input_acceleration >= -0.2278787921621627
 Relaxed_CLF_constraint: 0.0350624918937683 x1_input_acceleration
   - 0.1645717620849609 x2_input_acceleration
   + 0.0358048677444458 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2980079054832458 x1_input_acceleration
   - 0.0551280975341797 x2_input_acceleration
   + 0.3494852781295776 x3_input_acceleration >= -0.0228719446243894
 CBF_Constraint_sphere_low: 0.198008120059967 x1_input_acceleration
   - 0.4551277160644531 x2_input_acceleration
   + 0.5494850873947144 x3_input_acceleration >= -0.1064455787022262
 CBF_Constraint_sphere_right: 0.2980080842971802 x1_input_acceleration
   - 0.9051272869110107 x2_input_acceleration
   + 0.3494852781295776 x3_input_acceleration >= -0.2269260458616677
 Relaxed_CLF_constraint: 0.0350043773651123 x1_input_acceleration
   - 0.1635634899139404 x2_input_acceleration
   + 0.0357425212860107 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2978916764259338 x1_input_acceleration
   - 0.0531115531921387 x2_input_acceleration
   + 0.3493605852127075 x3_input_acceleration >= -0.022778276616625
 CBF_Constraint_sphere_low: 0.197891891002655 x1_input_acceleration
   - 0.4531111717224121 x2_input_acceleration
   + 0.5493603944778442 x3_input_acceleration >= -0.1059419443712791
 CBF_Constraint_sphere_right: 0.2978918552398682 x1_input_acceleration
   - 0.9031107425689697 x2_input_acceleration
   + 0.3493605852127075 x3_input_acceleration >= -0.225975347315473
 Relaxed_CLF_constraint: 0.0349462628364563 x1_input_acceleration
   - 0.1625552177429199 x2_input_acceleration
   + 0.0356801748275757 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2977660298347473 x1_input_acceleration
   - 0.0510993003845215 x2_input_acceleration
   + 0.3492265939712524 x3_input_acceleration >= -0.0226837402400243
 CBF_Constraint_sphere_low: 0.1977662444114685 x1_input_acceleration
   - 0.4510989189147949 x2_input_acceleration
   + 0.5492264032363892 x3_input_acceleration >= -0.1054378410216735
 CBF_Constraint_sphere_right: 0.2977662086486816 x1_input_acceleration
   - 0.9010984897613525 x2_input_acceleration
   + 0.3492265939712524 x3_input_acceleration >= -0.2250256042999907
 Relaxed_CLF_constraint: 0.034883439540863 x1_input_acceleration
   - 0.1615490913391113 x2_input_acceleration
   + 0.0356131792068481 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2975682616233826 x1_input_acceleration
   - 0.0491185188293457 x2_input_acceleration
   + 0.3490203619003296 x3_input_acceleration >= -0.0225686780596469
 CBF_Constraint_sphere_low: 0.1975684762001038 x1_input_acceleration
   - 0.4491181373596191 x2_input_acceleration
   + 0.5490201711654663 x3_input_acceleration >= -0.1049158881099907
 CBF_Constraint_sphere_right: 0.2975684404373169 x1_input_acceleration
   - 0.8991177082061768 x2_input_acceleration
   + 0.3490203619003296 x3_input_acceleration >= -0.2240687107438154
 Relaxed_CLF_constraint: 0.0347845554351807 x1_input_acceleration
   - 0.1605587005615234 x2_input_acceleration
   + 0.0355100631713867 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2973704934120178 x1_input_acceleration
   - 0.0471377372741699 x2_input_acceleration
   + 0.3488141298294067 x3_input_acceleration >= -0.0224556184490205
 CBF_Constraint_sphere_low: 0.197370707988739 x1_input_acceleration
   - 0.4471373558044434 x2_input_acceleration
   + 0.5488139390945435 x3_input_acceleration >= -0.1043959377680587
 CBF_Constraint_sphere_right: 0.2973706722259521 x1_input_acceleration
   - 0.897136926651001 x2_input_acceleration
   + 0.3488141298294067 x3_input_acceleration >= -0.2231138197573911
 Relaxed_CLF_constraint: 0.0346856713294983 x1_input_acceleration
   - 0.1595683097839355 x2_input_acceleration
   + 0.0354069471359253 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2971727252006531 x1_input_acceleration
   - 0.0451569557189941 x2_input_acceleration
   + 0.3486078977584839 x3_input_acceleration >= -0.0223445614081449
 CBF_Constraint_sphere_low: 0.1971729397773743 x1_input_acceleration
   - 0.4451565742492676 x2_input_acceleration
   + 0.5486077070236206 x3_input_acceleration >= -0.1038779899958776
 CBF_Constraint_sphere_right: 0.2971729040145874 x1_input_acceleration
   - 0.8951561450958252 x2_input_acceleration
   + 0.3486078977584839 x3_input_acceleration >= -0.2221609313407177
 Relaxed_CLF_constraint: 0.0345867872238159 x1_input_acceleration
   - 0.1585779190063477 x2_input_acceleration
   + 0.0353038311004639 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2969749569892883 x1_input_acceleration
   - 0.0431761741638184 x2_input_acceleration
   + 0.3484017848968506 x3_input_acceleration >= -0.0222355277033813
 CBF_Constraint_sphere_low: 0.1969751715660095 x1_input_acceleration
   - 0.4431757926940918 x2_input_acceleration
   + 0.5484015941619873 x3_input_acceleration >= -0.1033620774807261
 CBF_Constraint_sphere_right: 0.2969751358032227 x1_input_acceleration
   - 0.8931753635406494 x2_input_acceleration
   + 0.3484017848968506 x3_input_acceleration >= -0.2212100662601563
 Relaxed_CLF_constraint: 0.0344879031181335 x1_input_acceleration
   - 0.1575875282287598 x2_input_acceleration
   + 0.0352007746696472 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2967773079872131 x1_input_acceleration
   - 0.0411953926086426 x2_input_acceleration
   + 0.3481955528259277 x3_input_acceleration >= -0.0221284934790177
 CBF_Constraint_sphere_low: 0.1967775225639343 x1_input_acceleration
   - 0.441195011138916 x2_input_acceleration
   + 0.5481953620910645 x3_input_acceleration >= -0.1028481465646053
 CBF_Constraint_sphere_right: 0.2967774868011475 x1_input_acceleration
   - 0.8911945819854736 x2_input_acceleration
   + 0.3481955528259277 x3_input_acceleration >= -0.2202612006600055
 Relaxed_CLF_constraint: 0.0343890786170959 x1_input_acceleration
   - 0.1565971374511719 x2_input_acceleration
   + 0.0350976586341858 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2965795397758484 x1_input_acceleration
   - 0.0392146110534668 x2_input_acceleration
   + 0.3479893207550049 x3_input_acceleration >= -0.0220234441233146
 CBF_Constraint_sphere_low: 0.1965797543525696 x1_input_acceleration
   - 0.4392142295837402 x2_input_acceleration
   + 0.5479891300201416 x3_input_acceleration >= -0.1023362064775967
 CBF_Constraint_sphere_right: 0.2965797185897827 x1_input_acceleration
   - 0.8892138004302979 x2_input_acceleration
   + 0.3479893207550049 x3_input_acceleration >= -0.2193143199285045
 Relaxed_CLF_constraint: 0.0342901945114136 x1_input_acceleration
   - 0.155606746673584 x2_input_acceleration
   + 0.0349945425987244 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2963817715644836 x1_input_acceleration
   - 0.037233829498291 x2_input_acceleration
   + 0.347783088684082 x3_input_acceleration >= -0.0219203973373624
 CBF_Constraint_sphere_low: 0.1963819861412048 x1_input_acceleration
   - 0.4372334480285645 x2_input_acceleration
   + 0.5477828979492188 x3_input_acceleration >= -0.101826268960339
 CBF_Constraint_sphere_right: 0.296381950378418 x1_input_acceleration
   - 0.8872330188751221 x2_input_acceleration
   + 0.347783088684082 x3_input_acceleration >= -0.2183694417667545
 Relaxed_CLF_constraint: 0.0341913104057312 x1_input_acceleration
   - 0.1546163558959961 x2_input_acceleration
   + 0.0348914265632629 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2961840033531189 x1_input_acceleration
   - 0.0352530479431152 x2_input_acceleration
   + 0.3475768566131592 x3_input_acceleration >= -0.0218193531211611
 CBF_Constraint_sphere_low: 0.1961842179298401 x1_input_acceleration
   - 0.4352526664733887 x2_input_acceleration
   + 0.5475766658782959 x3_input_acceleration >= -0.1013183340128322
 CBF_Constraint_sphere_right: 0.2961841821670532 x1_input_acceleration
   - 0.8852522373199463 x2_input_acceleration
   + 0.3475768566131592 x3_input_acceleration >= -0.2174265661747553
 Relaxed_CLF_constraint: 0.0340924263000488 x1_input_acceleration
   - 0.1536259651184082 x2_input_acceleration
   + 0.0347883105278015 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.295985996723175 x1_input_acceleration
   - 0.0332725048065186 x2_input_acceleration
   + 0.3473703861236572 x3_input_acceleration >= -0.0217202387470025
 CBF_Constraint_sphere_low: 0.1959862112998962 x1_input_acceleration
   - 0.433272123336792 x2_input_acceleration
   + 0.5473701953887939 x3_input_acceleration >= -0.1008123646701066
 CBF_Constraint_sphere_right: 0.2959861755371094 x1_input_acceleration
   - 0.8832716941833496 x2_input_acceleration
   + 0.3473703861236572 x3_input_acceleration >= -0.216485721752577
 Relaxed_CLF_constraint: 0.0339934229850769 x1_input_acceleration
   - 0.1526356935501099 x2_input_acceleration
   + 0.0346850752830505 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2957882285118103 x1_input_acceleration
   - 0.0312917232513428 x2_input_acceleration
   + 0.3471641540527344 x3_input_acceleration >= -0.0216231994823361
 CBF_Constraint_sphere_low: 0.1957884430885315 x1_input_acceleration
   - 0.4312913417816162 x2_input_acceleration
   + 0.5471639633178711 x3_input_acceleration >= -0.1003084346741347
 CBF_Constraint_sphere_right: 0.2957884073257446 x1_input_acceleration
   - 0.8812909126281738 x2_input_acceleration
   + 0.3471641540527344 x3_input_acceleration >= -0.2155468511121127
 Relaxed_CLF_constraint: 0.0338945388793945 x1_input_acceleration
   - 0.151645302772522 x2_input_acceleration
   + 0.0345819592475891 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2955904603004456 x1_input_acceleration
   - 0.0293111801147461 x2_input_acceleration
   + 0.3469579219818115 x3_input_acceleration >= -0.0215281662815713
 CBF_Constraint_sphere_low: 0.1955906748771667 x1_input_acceleration
   - 0.4293107986450195 x2_input_acceleration
   + 0.5469577312469482 x3_input_acceleration >= -0.0998065584257347
 CBF_Constraint_sphere_right: 0.2955906391143799 x1_input_acceleration
   - 0.8793103694915771 x2_input_acceleration
   + 0.3469579219818115 x3_input_acceleration >= -0.2146100878633496
 Relaxed_CLF_constraint: 0.0337956547737122 x1_input_acceleration
   - 0.1506550312042236 x2_input_acceleration
   + 0.0344788432121277 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2953923344612122 x1_input_acceleration
   - 0.0273306369781494 x2_input_acceleration
   + 0.3467514514923096 x3_input_acceleration >= -0.0214350410220249
 CBF_Constraint_sphere_low: 0.1953925490379333 x1_input_acceleration
   - 0.4273302555084229 x2_input_acceleration
   + 0.5467512607574463 x3_input_acceleration >= -0.099306584158073
 CBF_Constraint_sphere_right: 0.2953925132751465 x1_input_acceleration
   - 0.8773298263549805 x2_input_acceleration
   + 0.3467514514923096 x3_input_acceleration >= -0.2136752325557728
 Relaxed_CLF_constraint: 0.0336965918540955 x1_input_acceleration
   - 0.1496647596359253 x2_input_acceleration
   + 0.0343756079673767 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2951863408088684 x1_input_acceleration
   - 0.0253536701202393 x2_input_acceleration
   + 0.3465369939804077 x3_input_acceleration >= -0.0213414181464181
 CBF_Constraint_sphere_low: 0.1951865553855896 x1_input_acceleration
   - 0.4253532886505127 x2_input_acceleration
   + 0.5465368032455444 x3_input_acceleration >= -0.0988064222177393
 CBF_Constraint_sphere_right: 0.2951865196228027 x1_input_acceleration
   - 0.8753528594970703 x2_input_acceleration
   + 0.3465369939804077 x3_input_acceleration >= -0.2127413995484246
 Relaxed_CLF_constraint: 0.0335935950279236 x1_input_acceleration
   - 0.1486762762069702 x2_input_acceleration
   + 0.0342683792114258 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.294933021068573 x1_input_acceleration
   - 0.0233986377716064 x2_input_acceleration
   + 0.3462752103805542 x3_input_acceleration >= -0.0212348761225748
 CBF_Constraint_sphere_low: 0.1949332356452942 x1_input_acceleration
   - 0.4233982563018799 x2_input_acceleration
   + 0.5462750196456909 x3_input_acceleration >= -0.098295361721879
 CBF_Constraint_sphere_right: 0.2949331998825073 x1_input_acceleration
   - 0.8733978271484375 x2_input_acceleration
   + 0.3462752103805542 x3_input_acceleration >= -0.211803969546161
 Relaxed_CLF_constraint: 0.0334669351577759 x1_input_acceleration
   - 0.1476987600326538 x2_input_acceleration
   + 0.034137487411499 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2946541905403137 x1_input_acceleration
   - 0.0214552879333496 x2_input_acceleration
   + 0.3459877967834473 x3_input_acceleration >= -0.0211222442265836
 CBF_Constraint_sphere_low: 0.1946544051170349 x1_input_acceleration
   - 0.421454906463623 x2_input_acceleration
   + 0.545987606048584 x3_input_acceleration >= -0.0977792603930979
 CBF_Constraint_sphere_right: 0.294654369354248 x1_input_acceleration
   - 0.8714544773101807 x2_input_acceleration
   + 0.3459877967834473 x3_input_acceleration >= -0.2108654147316435
 Relaxed_CLF_constraint: 0.0333275198936462 x1_input_acceleration
   - 0.1467270851135254 x2_input_acceleration
   + 0.0339937806129456 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2943753600120544 x1_input_acceleration
   - 0.0195119380950928 x2_input_acceleration
   + 0.3457003831863403 x3_input_acceleration >= -0.021011580811409
 CBF_Constraint_sphere_low: 0.1943755745887756 x1_input_acceleration
   - 0.4195115566253662 x2_input_acceleration
   + 0.5457001924514771 x3_input_acceleration >= -0.0972651275451334
 CBF_Constraint_sphere_right: 0.2943755388259888 x1_input_acceleration
   - 0.8695111274719238 x2_input_acceleration
   + 0.3457003831863403 x3_input_acceleration >= -0.2099288283979425
 Relaxed_CLF_constraint: 0.0331881046295166 x1_input_acceleration
   - 0.145755410194397 x2_input_acceleration
   + 0.0338500738143921 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2940964102745056 x1_input_acceleration
   - 0.0175685882568359 x2_input_acceleration
   + 0.3454129695892334 x3_input_acceleration >= -0.0209028683475353
 CBF_Constraint_sphere_low: 0.1940966248512268 x1_input_acceleration
   - 0.4175682067871094 x2_input_acceleration
   + 0.5454127788543701 x3_input_acceleration >= -0.0967529516089215
 CBF_Constraint_sphere_right: 0.2940965890884399 x1_input_acceleration
   - 0.867567777633667 x2_input_acceleration
   + 0.3454129695892334 x3_input_acceleration >= -0.2089941930155318
 Relaxed_CLF_constraint: 0.0330486297607422 x1_input_acceleration
   - 0.1447837352752686 x2_input_acceleration
   + 0.0337063670158386 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2938174605369568 x1_input_acceleration
   - 0.0156252384185791 x2_input_acceleration
   + 0.3451255559921265 x3_input_acceleration >= -0.0207961243977245
 CBF_Constraint_sphere_low: 0.193817675113678 x1_input_acceleration
   - 0.4156248569488525 x2_input_acceleration
   + 0.5451253652572632 x3_input_acceleration >= -0.0962427441867725
 CBF_Constraint_sphere_right: 0.2938176393508911 x1_input_acceleration
   - 0.8656244277954102 x2_input_acceleration
   + 0.3451255559921265 x3_input_acceleration >= -0.208061526147184
 Relaxed_CLF_constraint: 0.0329091548919678 x1_input_acceleration
   - 0.1438120603561401 x2_input_acceleration
   + 0.0335626602172852 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2935386300086975 x1_input_acceleration
   - 0.0136818885803223 x2_input_acceleration
   + 0.3448381423950195 x3_input_acceleration >= -0.0206913664582388
 CBF_Constraint_sphere_low: 0.1935388445854187 x1_input_acceleration
   - 0.4136815071105957 x2_input_acceleration
   + 0.5448379516601562 x3_input_acceleration >= -0.0957345168144969
 CBF_Constraint_sphere_right: 0.2935388088226318 x1_input_acceleration
   - 0.8636810779571533 x2_input_acceleration
   + 0.3448381423950195 x3_input_acceleration >= -0.2071308452891719
 Relaxed_CLF_constraint: 0.0327697396278381 x1_input_acceleration
   - 0.1428403854370117 x2_input_acceleration
   + 0.0334189534187317 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2932596802711487 x1_input_acceleration
   - 0.0117385387420654 x2_input_acceleration
   + 0.3445507287979126 x3_input_acceleration >= -0.020588559519927
 CBF_Constraint_sphere_low: 0.1932598948478699 x1_input_acceleration
   - 0.4117381572723389 x2_input_acceleration
   + 0.5445505380630493 x3_input_acceleration >= -0.0952282464038469
 CBF_Constraint_sphere_right: 0.293259859085083 x1_input_acceleration
   - 0.8617377281188965 x2_input_acceleration
   + 0.3445507287979126 x3_input_acceleration >= -0.2062021154323232
 Relaxed_CLF_constraint: 0.0326302647590637 x1_input_acceleration
   - 0.1418687105178833 x2_input_acceleration
   + 0.0332752466201782 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2929807305335999 x1_input_acceleration
   - 0.00979518890380859 x2_input_acceleration
   + 0.3442633152008057 x3_input_acceleration >= -0.0204877210956781
 CBF_Constraint_sphere_low: 0.192980945110321 x1_input_acceleration
   - 0.409794807434082 x2_input_acceleration
   + 0.5442631244659424 x3_input_acceleration >= -0.0947239445072598
 CBF_Constraint_sphere_right: 0.2929809093475342 x1_input_acceleration
   - 0.8597943782806396 x2_input_acceleration
   + 0.3442633152008057 x3_input_acceleration >= -0.2052753540895373
 Relaxed_CLF_constraint: 0.0324907898902893 x1_input_acceleration
   - 0.1408970355987549 x2_input_acceleration
   + 0.0331315398216248 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.292701780796051 x1_input_acceleration
   - 0.00785183906555176 x2_input_acceleration
   + 0.3439759016036987 x3_input_acceleration >= -0.0203888511854921
 CBF_Constraint_sphere_low: 0.1927019953727722 x1_input_acceleration
   - 0.4078514575958252 x2_input_acceleration
   + 0.5439757108688354 x3_input_acceleration >= -0.0942216111247355
 CBF_Constraint_sphere_right: 0.2927019596099854 x1_input_acceleration
   - 0.8578510284423828 x2_input_acceleration
   + 0.3439759016036987 x3_input_acceleration >= -0.2043505612608142
 Relaxed_CLF_constraint: 0.0323513150215149 x1_input_acceleration
   - 0.1399253606796265 x2_input_acceleration
   + 0.0329878330230713 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2924228310585022 x1_input_acceleration
   - 0.00590848922729492 x2_input_acceleration
   + 0.3436884880065918 x3_input_acceleration >= -0.0202919497893689
 CBF_Constraint_sphere_low: 0.1924230456352234 x1_input_acceleration
   - 0.4059081077575684 x2_input_acceleration
   + 0.5436882972717285 x3_input_acceleration >= -0.0937212462562741
 CBF_Constraint_sphere_right: 0.2924230098724365 x1_input_acceleration
   - 0.855907678604126 x2_input_acceleration
   + 0.3436884880065918 x3_input_acceleration >= -0.203427736946154
 Relaxed_CLF_constraint: 0.0322118401527405 x1_input_acceleration
   - 0.138953685760498 x2_input_acceleration
   + 0.0328441262245178 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2921438813209534 x1_input_acceleration
   - 0.00396513938903809 x2_input_acceleration
   + 0.3434011936187744 x3_input_acceleration >= -0.0201970373756112
 CBF_Constraint_sphere_low: 0.1921440958976746 x1_input_acceleration
   - 0.4039647579193115 x2_input_acceleration
   + 0.5434010028839111 x3_input_acceleration >= -0.0932228822910958
 CBF_Constraint_sphere_right: 0.2921440601348877 x1_input_acceleration
   - 0.8539643287658691 x2_input_acceleration
   + 0.3434011936187744 x3_input_acceleration >= -0.2025069016138593
 Relaxed_CLF_constraint: 0.0320723652839661 x1_input_acceleration
   - 0.1379820108413696 x2_input_acceleration
   + 0.0327004790306091 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.291864812374115 x1_input_acceleration
   - 0.00202178955078125 x2_input_acceleration
   + 0.3431136608123779 x3_input_acceleration >= -0.0201040351428091
 CBF_Constraint_sphere_low: 0.1918650269508362 x1_input_acceleration
   - 0.4020214080810547 x2_input_acceleration
   + 0.5431134700775146 x3_input_acceleration >= -0.0927264106254896
 CBF_Constraint_sphere_right: 0.2918649911880493 x1_input_acceleration
   - 0.8520209789276123 x2_input_acceleration
   + 0.3431136608123779 x3_input_acceleration >= -0.2015879764625096
 Relaxed_CLF_constraint: 0.0319328308105469 x1_input_acceleration
   - 0.1370103359222412 x2_input_acceleration
   + 0.0325567126274109 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2915859818458557 x1_input_acceleration
   - 7.84397e-05 x2_input_acceleration
   + 0.342826247215271 x3_input_acceleration >= -0.0200130566853766
 CBF_Constraint_sphere_low: 0.1915861964225769 x1_input_acceleration
   - 0.4000780582427979 x2_input_acceleration
   + 0.5428260564804077 x3_input_acceleration >= -0.0922319627352671
 CBF_Constraint_sphere_right: 0.29158616065979 x1_input_acceleration
   - 0.8500776290893555 x2_input_acceleration
   + 0.342826247215271 x3_input_acceleration >= -0.2006710750865507
 Relaxed_CLF_constraint: 0.0317934155464172 x1_input_acceleration
   - 0.1360386610031128 x2_input_acceleration
   + 0.0324130058288574 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2913071513175964 x1_input_acceleration
   + 0.00186491012573242 x2_input_acceleration
   + 0.3425388336181641 x3_input_acceleration >= -0.0199240467087606
 CBF_Constraint_sphere_low: 0.1913073658943176 x1_input_acceleration
   - 0.398134708404541 x2_input_acceleration
   + 0.5425386428833008 x3_input_acceleration >= -0.0917394833258613
 CBF_Constraint_sphere_right: 0.2913073301315308 x1_input_acceleration
   - 0.8481342792510986 x2_input_acceleration
   + 0.3425388336181641 x3_input_acceleration >= -0.1997561421914084
 Relaxed_CLF_constraint: 0.0316540002822876 x1_input_acceleration
   - 0.1350669860839844 x2_input_acceleration
   + 0.032269299030304 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2910270094871521 x1_input_acceleration
   + 0.00380778312683105 x2_input_acceleration
   + 0.3422502279281616 x3_input_acceleration >= -0.0198366094950636
 CBF_Constraint_sphere_low: 0.1910272240638733 x1_input_acceleration
   - 0.3961918354034424 x2_input_acceleration
   + 0.5422500371932983 x3_input_acceleration >= -0.0912486184025078
 CBF_Constraint_sphere_right: 0.2910271883010864 x1_input_acceleration
   - 0.84619140625 x2_input_acceleration
   + 0.3422502279281616 x3_input_acceleration >= -0.1988429847146668
 Relaxed_CLF_constraint: 0.0315139293670654 x1_input_acceleration
   - 0.1340955495834351 x2_input_acceleration
   + 0.0321249961853027 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2907455563545227 x1_input_acceleration
   + 0.00574994087219238 x2_input_acceleration
   + 0.3419604301452637 x3_input_acceleration >= -0.0197507440362671
 CBF_Constraint_sphere_low: 0.1907457709312439 x1_input_acceleration
   - 0.3942496776580811 x2_input_acceleration
   + 0.5419602394104004 x3_input_acceleration >= -0.0907594146408585
 CBF_Constraint_sphere_right: 0.290745735168457 x1_input_acceleration
   - 0.8442492485046387 x2_input_acceleration
   + 0.3419604301452637 x3_input_acceleration >= -0.1979317029761069
 Relaxed_CLF_constraint: 0.0313732028007507 x1_input_acceleration
   - 0.1331244707107544 x2_input_acceleration
   + 0.0319800972938538 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2904649376869202 x1_input_acceleration
   + 0.00769257545471191 x2_input_acceleration
   + 0.341671347618103 x3_input_acceleration >= -0.0196670913814409
 CBF_Constraint_sphere_low: 0.1904651522636414 x1_input_acceleration
   - 0.3923070430755615 x2_input_acceleration
   + 0.5416711568832397 x3_input_acceleration >= -0.0902723581181825
 CBF_Constraint_sphere_right: 0.2904651165008545 x1_input_acceleration
   - 0.8423066139221191 x2_input_acceleration
   + 0.341671347618103 x3_input_acceleration >= -0.1970224313859929
 Relaxed_CLF_constraint: 0.0312328934669495 x1_input_acceleration
   - 0.1321531534194946 x2_input_acceleration
   + 0.0318355560302734 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2901601195335388 x1_input_acceleration
   + 0.00962352752685547 x2_input_acceleration
   + 0.3413581848144531 x3_input_acceleration >= -0.019577729397399
 CBF_Constraint_sphere_low: 0.19016033411026 x1_input_acceleration
   - 0.390376091003418 x2_input_acceleration
   + 0.5413579940795898 x3_input_acceleration >= -0.089780730712478
 CBF_Constraint_sphere_right: 0.2901602983474731 x1_input_acceleration
   - 0.8403756618499756 x2_input_acceleration
   + 0.3413581848144531 x3_input_acceleration >= -0.1961124155266744
 Relaxed_CLF_constraint: 0.0310804843902588 x1_input_acceleration
   - 0.1311876773834229 x2_input_acceleration
   + 0.0316789746284485 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2898357510566711 x1_input_acceleration
   + 0.0115454196929932 x2_input_acceleration
   + 0.3410263061523438 x3_input_acceleration >= -0.019484250198596
 CBF_Constraint_sphere_low: 0.1898359656333923 x1_input_acceleration
   - 0.3884541988372803 x2_input_acceleration
   + 0.5410261154174805 x3_input_acceleration >= -0.0892859040015012
 CBF_Constraint_sphere_right: 0.2898359298706055 x1_input_acceleration
   - 0.8384537696838379 x2_input_acceleration
   + 0.3410263061523438 x3_input_acceleration >= -0.1952021329072272
 Relaxed_CLF_constraint: 0.030918300151825 x1_input_acceleration
   - 0.130226731300354 x2_input_acceleration
   + 0.0315130352973938 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2895107865333557 x1_input_acceleration
   + 0.0134673118591309 x2_input_acceleration
   + 0.3406949043273926 x3_input_acceleration >= -0.0193927204606311
 CBF_Constraint_sphere_low: 0.1895110011100769 x1_input_acceleration
   - 0.3865323066711426 x2_input_acceleration
   + 0.5406947135925293 x3_input_acceleration >= -0.0887931042372913
 CBF_Constraint_sphere_right: 0.28951096534729 x1_input_acceleration
   - 0.8365318775177002 x2_input_acceleration
   + 0.3406949043273926 x3_input_acceleration >= -0.1942937997485649
 Relaxed_CLF_constraint: 0.0307558178901672 x1_input_acceleration
   - 0.1292657852172852 x2_input_acceleration
   + 0.0313473343849182 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2891855835914612 x1_input_acceleration
   + 0.0153892040252686 x2_input_acceleration
   + 0.3403637409210205 x3_input_acceleration >= -0.0193031513728542
 CBF_Constraint_sphere_low: 0.1891857981681824 x1_input_acceleration
   - 0.3846104145050049 x2_input_acceleration
   + 0.5403635501861572 x3_input_acceleration >= -0.0883023008860079
 CBF_Constraint_sphere_right: 0.2891857624053955 x1_input_acceleration
   - 0.8346099853515625 x2_input_acceleration
   + 0.3403637409210205 x3_input_acceleration >= -0.1933874272400692
 Relaxed_CLF_constraint: 0.03059321641922 x1_input_acceleration
   - 0.1283048391342163 x2_input_acceleration
   + 0.0311817526817322 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2888602614402771 x1_input_acceleration
   + 0.0173110961914062 x2_input_acceleration
   + 0.340032696723938 x3_input_acceleration >= -0.0192155398830117
 CBF_Constraint_sphere_low: 0.1888604760169983 x1_input_acceleration
   - 0.3826885223388672 x2_input_acceleration
   + 0.5400325059890747 x3_input_acceleration >= -0.0878134730140282
 CBF_Constraint_sphere_right: 0.2888604402542114 x1_input_acceleration
   - 0.8326880931854248 x2_input_acceleration
   + 0.340032696723938 x3_input_acceleration >= -0.1924830123294973
 Relaxed_CLF_constraint: 0.0304305553436279 x1_input_acceleration
   - 0.1273438930511475 x2_input_acceleration
   + 0.0310162305831909 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2885348200798035 x1_input_acceleration
   + 0.0192329883575439 x2_input_acceleration
   + 0.339701771736145 x3_input_acceleration >= -0.0191298859900805
 CBF_Constraint_sphere_low: 0.1885350346565247 x1_input_acceleration
   - 0.3807666301727295 x2_input_acceleration
   + 0.5397015810012817 x3_input_acceleration >= -0.087326620620329
 CBF_Constraint_sphere_right: 0.2885349988937378 x1_input_acceleration
   - 0.8307662010192871 x2_input_acceleration
   + 0.339701771736145 x3_input_acceleration >= -0.191580555015826
 Relaxed_CLF_constraint: 0.0302678346633911 x1_input_acceleration
   - 0.1263829469680786 x2_input_acceleration
   + 0.0308507680892944 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2882093787193298 x1_input_acceleration
   + 0.0211548805236816 x2_input_acceleration
   + 0.3393709659576416 x3_input_acceleration >= -0.019046206871694
 CBF_Constraint_sphere_low: 0.188209593296051 x1_input_acceleration
   - 0.3788447380065918 x2_input_acceleration
   + 0.5393707752227783 x3_input_acceleration >= -0.0868417549220922
 CBF_Constraint_sphere_right: 0.2882095575332642 x1_input_acceleration
   - 0.8288443088531494 x2_input_acceleration
   + 0.3393709659576416 x3_input_acceleration >= -0.1906800724766994
 Relaxed_CLF_constraint: 0.0301051139831543 x1_input_acceleration
   - 0.1254220008850098 x2_input_acceleration
   + 0.0306853652000427 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2878838181495667 x1_input_acceleration
   + 0.0230767726898193 x2_input_acceleration
   + 0.3390401601791382 x3_input_acceleration >= -0.0189644651011115
 CBF_Constraint_sphere_low: 0.1878840327262878 x1_input_acceleration
   - 0.3769228458404541 x2_input_acceleration
   + 0.5390399694442749 x3_input_acceleration >= -0.0863588325321111
 CBF_Constraint_sphere_right: 0.287883996963501 x1_input_acceleration
   - 0.8269224166870117 x2_input_acceleration
   + 0.3390401601791382 x3_input_acceleration >= -0.1897815272853662
 Relaxed_CLF_constraint: 0.0299423336982727 x1_input_acceleration
   - 0.1244610548019409 x2_input_acceleration
   + 0.030519962310791 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2875582575798035 x1_input_acceleration
   + 0.0249984264373779 x2_input_acceleration
   + 0.3387093544006348 x3_input_acceleration >= -0.0188846748962931
 CBF_Constraint_sphere_low: 0.1875584721565247 x1_input_acceleration
   - 0.3750011920928955 x2_input_acceleration
   + 0.5387091636657715 x3_input_acceleration >= -0.0858779093915644
 CBF_Constraint_sphere_right: 0.2875584363937378 x1_input_acceleration
   - 0.8250007629394531 x2_input_acceleration
   + 0.3387093544006348 x3_input_acceleration >= -0.1888850349875965
 Relaxed_CLF_constraint: 0.0297795534133911 x1_input_acceleration
   - 0.1235002279281616 x2_input_acceleration
   + 0.0303545594215393 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2872326970100403 x1_input_acceleration
   + 0.0269203186035156 x2_input_acceleration
   + 0.3383785486221313 x3_input_acceleration >= -0.0188068419882491
 CBF_Constraint_sphere_low: 0.1872329115867615 x1_input_acceleration
   - 0.3730792999267578 x2_input_acceleration
   + 0.5383783578872681 x3_input_acceleration >= -0.0853988958641218
 CBF_Constraint_sphere_right: 0.2872328758239746 x1_input_acceleration
   - 0.8230788707733154 x2_input_acceleration
   + 0.3383785486221313 x3_input_acceleration >= -0.1879903986588018
 Relaxed_CLF_constraint: 0.0296167731285095 x1_input_acceleration
   - 0.1225392818450928 x2_input_acceleration
   + 0.0301891565322876 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2869071364402771 x1_input_acceleration
   + 0.0288422107696533 x2_input_acceleration
   + 0.3380477428436279 x3_input_acceleration >= -0.0187309636260281
 CBF_Constraint_sphere_low: 0.1869073510169983 x1_input_acceleration
   - 0.3711574077606201 x2_input_acceleration
   + 0.5380475521087646 x3_input_acceleration >= -0.0849218368825021
 CBF_Constraint_sphere_right: 0.2869073152542114 x1_input_acceleration
   - 0.8211569786071777 x2_input_acceleration
   + 0.3380477428436279 x3_input_acceleration >= -0.18709771687583
 Relaxed_CLF_constraint: 0.0294539928436279 x1_input_acceleration
   - 0.1215783357620239 x2_input_acceleration
   + 0.0300237536430359 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2865815758705139 x1_input_acceleration
   + 0.030764102935791 x2_input_acceleration
   + 0.3377169370651245 x3_input_acceleration >= -0.0186570398096301
 CBF_Constraint_sphere_low: 0.1865817904472351 x1_input_acceleration
   - 0.3692355155944824 x2_input_acceleration
   + 0.5377167463302612 x3_input_acceleration >= -0.0844467324467054
 CBF_Constraint_sphere_right: 0.2865817546844482 x1_input_acceleration
   - 0.81923508644104 x2_input_acceleration
   + 0.3377169370651245 x3_input_acceleration >= -0.1862069896386812
 Relaxed_CLF_constraint: 0.0292912125587463 x1_input_acceleration
   - 0.1206173896789551 x2_input_acceleration
   + 0.0298583507537842 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2862560153007507 x1_input_acceleration
   + 0.0326859951019287 x2_input_acceleration
   + 0.3373861312866211 x3_input_acceleration >= -0.018585070539055
 CBF_Constraint_sphere_low: 0.1862562298774719 x1_input_acceleration
   - 0.3673136234283447 x2_input_acceleration
   + 0.5373859405517578 x3_input_acceleration >= -0.0839735825567317
 CBF_Constraint_sphere_right: 0.2862561941146851 x1_input_acceleration
   - 0.8173131942749023 x2_input_acceleration
   + 0.3373861312866211 x3_input_acceleration >= -0.1853182169473554
 Relaxed_CLF_constraint: 0.0291284322738647 x1_input_acceleration
   - 0.1196564435958862 x2_input_acceleration
   + 0.0296929478645325 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2859304547309875 x1_input_acceleration
   + 0.0346078872680664 x2_input_acceleration
   + 0.3370553255081177 x3_input_acceleration >= -0.0185150558143029
 CBF_Constraint_sphere_low: 0.1859306693077087 x1_input_acceleration
   - 0.365391731262207 x2_input_acceleration
   + 0.5370551347732544 x3_input_acceleration >= -0.083502387212581
 CBF_Constraint_sphere_right: 0.2859306335449219 x1_input_acceleration
   - 0.8153913021087646 x2_input_acceleration
   + 0.3370553255081177 x3_input_acceleration >= -0.1844313988018526
 Relaxed_CLF_constraint: 0.0289656519889832 x1_input_acceleration
   - 0.1186954975128174 x2_input_acceleration
   + 0.0295275449752808 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2856048941612244 x1_input_acceleration
   + 0.0365297794342041 x2_input_acceleration
   + 0.3367246389389038 x3_input_acceleration >= -0.0184470157057227
 CBF_Constraint_sphere_low: 0.1856051087379456 x1_input_acceleration
   - 0.3634698390960693 x2_input_acceleration
   + 0.5367244482040405 x3_input_acceleration >= -0.0830331784055197
 CBF_Constraint_sphere_right: 0.2856050729751587 x1_input_acceleration
   - 0.813469409942627 x2_input_acceleration
   + 0.3367246389389038 x3_input_acceleration >= -0.1835465552725216
 Relaxed_CLF_constraint: 0.0288028717041016 x1_input_acceleration
   - 0.1177345514297485 x2_input_acceleration
   + 0.0293622016906738 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2852761149406433 x1_input_acceleration
   + 0.0384500026702881 x2_input_acceleration
   + 0.336390495300293 x3_input_acceleration >= -0.0183798574473372
 CBF_Constraint_sphere_low: 0.1852763295173645 x1_input_acceleration
   - 0.3615496158599854 x2_input_acceleration
   + 0.5363903045654297 x3_input_acceleration >= -0.0825650004599311
 CBF_Constraint_sphere_right: 0.2852762937545776 x1_input_acceleration
   - 0.811549186706543 x2_input_acceleration
   + 0.336390495300293 x3_input_acceleration >= -0.182663302887694
 Relaxed_CLF_constraint: 0.028638482093811 x1_input_acceleration
   - 0.1167744398117065 x2_input_acceleration
   + 0.0291951298713684 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.284950315952301 x1_input_acceleration
   + 0.0403716564178467 x2_input_acceleration
   + 0.3360594511032104 x3_input_acceleration >= -0.018315626969757
 CBF_Constraint_sphere_low: 0.1849505305290222 x1_input_acceleration
   - 0.3596279621124268 x2_input_acceleration
   + 0.5360592603683472 x3_input_acceleration >= -0.0820996251256908
 CBF_Constraint_sphere_right: 0.2849504947662354 x1_input_acceleration
   - 0.8096275329589844 x2_input_acceleration
   + 0.3360594511032104 x3_input_acceleration >= -0.1817823703171412
 Relaxed_CLF_constraint: 0.0284755825996399 x1_input_acceleration
   - 0.1158136129379272 x2_input_acceleration
   + 0.0290296077728271 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2846236824989319 x1_input_acceleration
   + 0.0422928333282471 x2_input_acceleration
   + 0.3357275724411011 x3_input_acceleration >= -0.0182530818218446
 CBF_Constraint_sphere_low: 0.1846238970756531 x1_input_acceleration
   - 0.3577067852020264 x2_input_acceleration
   + 0.5357273817062378 x3_input_acceleration >= -0.0816359887651977
 CBF_Constraint_sphere_right: 0.2846238613128662 x1_input_acceleration
   - 0.807706356048584 x2_input_acceleration
   + 0.3357275724411011 x3_input_acceleration >= -0.1809033257317806
 Relaxed_CLF_constraint: 0.0283122658729553 x1_input_acceleration
   - 0.1148530244827271 x2_input_acceleration
   + 0.0288636684417725 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2842665314674377 x1_input_acceleration
   + 0.0441985130310059 x2_input_acceleration
   + 0.3353643417358398 x3_input_acceleration >= -0.0181825527936482
 CBF_Constraint_sphere_low: 0.1842667460441589 x1_input_acceleration
   - 0.3558011054992676 x2_input_acceleration
   + 0.5353641510009766 x3_input_acceleration >= -0.0811658586373
 CBF_Constraint_sphere_right: 0.2842667102813721 x1_input_acceleration
   - 0.8058006763458252 x2_input_acceleration
   + 0.3353643417358398 x3_input_acceleration >= -0.180022883570374
 Relaxed_CLF_constraint: 0.0281336903572083 x1_input_acceleration
   - 0.1139001846313477 x2_input_acceleration
   + 0.0286820530891418 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2838733792304993 x1_input_acceleration
   + 0.0460860729217529 x2_input_acceleration
   + 0.3349640369415283 x3_input_acceleration >= -0.0181022318993144
 CBF_Constraint_sphere_low: 0.1838735938072205 x1_input_acceleration
   - 0.3539135456085205 x2_input_acceleration
   + 0.534963846206665 x3_input_acceleration >= -0.0806876532532513
 CBF_Constraint_sphere_right: 0.2838735580444336 x1_input_acceleration
   - 0.8039131164550781 x2_input_acceleration
   + 0.3349640369415283 x3_input_acceleration >= -0.179140350452372
 Relaxed_CLF_constraint: 0.027937114238739 x1_input_acceleration
   - 0.1129564046859741 x2_input_acceleration
   + 0.0284819006919861 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2834693789482117 x1_input_acceleration
   + 0.0479681491851807 x2_input_acceleration
   + 0.3345526456832886 x3_input_acceleration >= -0.0180203262178062
 CBF_Constraint_sphere_low: 0.1834695935249329 x1_input_acceleration
   - 0.3520314693450928 x2_input_acceleration
   + 0.5345524549484253 x3_input_acceleration >= -0.0802083935622143
 CBF_Constraint_sphere_right: 0.283469557762146 x1_input_acceleration
   - 0.8020310401916504 x2_input_acceleration
   + 0.3345526456832886 x3_input_acceleration >= -0.1782585630856139
 Relaxed_CLF_constraint: 0.0277351140975952 x1_input_acceleration
   - 0.1120153665542603 x2_input_acceleration
   + 0.0282762050628662 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2830654978752136 x1_input_acceleration
   + 0.0498502254486084 x2_input_acceleration
   + 0.3341412544250488 x3_input_acceleration >= -0.0179403747433412
 CBF_Constraint_sphere_low: 0.1830657124519348 x1_input_acceleration
   - 0.350149393081665 x2_input_acceleration
   + 0.5341410636901855 x3_input_acceleration >= -0.0797310821177689
 CBF_Constraint_sphere_right: 0.2830656766891479 x1_input_acceleration
   - 0.8001489639282227 x2_input_acceleration
   + 0.3341412544250488 x3_input_acceleration >= -0.1773787299259097
 Relaxed_CLF_constraint: 0.0275331735610962 x1_input_acceleration
   - 0.1110743284225464 x2_input_acceleration
   + 0.0280705094337463 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.282661497592926 x1_input_acceleration
   + 0.0517323017120361 x2_input_acceleration
   + 0.3337298631668091 x3_input_acceleration >= -0.0178623437078095
 CBF_Constraint_sphere_low: 0.1826617121696472 x1_input_acceleration
   - 0.3482673168182373 x2_input_acceleration
   + 0.5337296724319458 x3_input_acceleration >= -0.0792556970727084
 CBF_Constraint_sphere_right: 0.2826616764068604 x1_input_acceleration
   - 0.7982668876647949 x2_input_acceleration
   + 0.3337298631668091 x3_input_acceleration >= -0.176500817205128
 Relaxed_CLF_constraint: 0.0273311734199524 x1_input_acceleration
   - 0.1101332902908325 x2_input_acceleration
   + 0.0278648138046265 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2822574973106384 x1_input_acceleration
   + 0.0536143779754639 x2_input_acceleration
   + 0.3333184719085693 x3_input_acceleration >= -0.0177862500073062
 CBF_Constraint_sphere_low: 0.1822577118873596 x1_input_acceleration
   - 0.3463852405548096 x2_input_acceleration
   + 0.5333182811737061 x3_input_acceleration >= -0.0787822493626763
 CBF_Constraint_sphere_right: 0.2822576761245728 x1_input_acceleration
   - 0.7963848114013672 x2_input_acceleration
   + 0.3333184719085693 x3_input_acceleration >= -0.1756248418193748
 Relaxed_CLF_constraint: 0.0271291732788086 x1_input_acceleration
   - 0.1091922521591187 x2_input_acceleration
   + 0.0276591181755066 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2818534970283508 x1_input_acceleration
   + 0.0554964542388916 x2_input_acceleration
   + 0.3329070806503296 x3_input_acceleration >= -0.0177120936418313
 CBF_Constraint_sphere_low: 0.181853711605072 x1_input_acceleration
   - 0.3445031642913818 x2_input_acceleration
   + 0.5329068899154663 x3_input_acceleration >= -0.0783107389876726
 CBF_Constraint_sphere_right: 0.2818536758422852 x1_input_acceleration
   - 0.7945027351379395 x2_input_acceleration
   + 0.3329070806503296 x3_input_acceleration >= -0.17475080376865
 Relaxed_CLF_constraint: 0.0269271731376648 x1_input_acceleration
   - 0.1082512140274048 x2_input_acceleration
   + 0.0274534225463867 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2814494967460632 x1_input_acceleration
   + 0.0573785305023193 x2_input_acceleration
   + 0.3324956893920898 x3_input_acceleration >= -0.0176398746113847
 CBF_Constraint_sphere_low: 0.1814497113227844 x1_input_acceleration
   - 0.3426210880279541 x2_input_acceleration
   + 0.5324954986572266 x3_input_acceleration >= -0.0778411659476973
 CBF_Constraint_sphere_right: 0.2814496755599976 x1_input_acceleration
   - 0.7926206588745117 x2_input_acceleration
   + 0.3324956893920898 x3_input_acceleration >= -0.1738787030529535
 Relaxed_CLF_constraint: 0.026725172996521 x1_input_acceleration
   - 0.1073101758956909 x2_input_acceleration
   + 0.0272477269172668 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2810456156730652 x1_input_acceleration
   + 0.0592606067657471 x2_input_acceleration
   + 0.3320842981338501 x3_input_acceleration >= -0.0175696096675872
 CBF_Constraint_sphere_low: 0.1810458302497864 x1_input_acceleration
   - 0.3407390117645264 x2_input_acceleration
   + 0.5320841073989868 x3_input_acceleration >= -0.0773735410339193
 CBF_Constraint_sphere_right: 0.2810457944869995 x1_input_acceleration
   - 0.790738582611084 x2_input_acceleration
   + 0.3320842981338501 x3_input_acceleration >= -0.1730085564239167
 Relaxed_CLF_constraint: 0.026523232460022 x1_input_acceleration
   - 0.1063691377639771 x2_input_acceleration
   + 0.027042031288147 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2806416153907776 x1_input_acceleration
   + 0.0611426830291748 x2_input_acceleration
   + 0.3316729068756104 x3_input_acceleration >= -0.0175012652831171
 CBF_Constraint_sphere_low: 0.1806418299674988 x1_input_acceleration
   - 0.3388569355010986 x2_input_acceleration
   + 0.5316727161407471 x3_input_acceleration >= -0.0769078426399205
 CBF_Constraint_sphere_right: 0.2806417942047119 x1_input_acceleration
   - 0.7888565063476562 x2_input_acceleration
   + 0.3316729068756104 x3_input_acceleration >= -0.1721403303541968
 Relaxed_CLF_constraint: 0.0263212323188782 x1_input_acceleration
   - 0.1054280996322632 x2_input_acceleration
   + 0.0268363356590271 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.28023761510849 x1_input_acceleration
   + 0.0630247592926025 x2_input_acceleration
   + 0.3312616348266602 x3_input_acceleration >= -0.0174348779784041
 CBF_Constraint_sphere_low: 0.1802378296852112 x1_input_acceleration
   - 0.3369748592376709 x2_input_acceleration
   + 0.5312614440917969 x3_input_acceleration >= -0.0764441132465962
 CBF_Constraint_sphere_right: 0.2802377939224243 x1_input_acceleration
   - 0.7869744300842285 x2_input_acceleration
   + 0.3312616348266602 x3_input_acceleration >= -0.1712740613642338
 Relaxed_CLF_constraint: 0.0261192321777344 x1_input_acceleration
   - 0.1044870615005493 x2_input_acceleration
   + 0.026630699634552 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2798337340354919 x1_input_acceleration
   + 0.0649068355560303 x2_input_acceleration
   + 0.3308502435684204 x3_input_acceleration >= -0.0173704249188568
 CBF_Constraint_sphere_low: 0.1798339486122131 x1_input_acceleration
   - 0.3350927829742432 x2_input_acceleration
   + 0.5308500528335571 x3_input_acceleration >= -0.0759823002170684
 CBF_Constraint_sphere_right: 0.2798339128494263 x1_input_acceleration
   - 0.7850923538208008 x2_input_acceleration
   + 0.3308502435684204 x3_input_acceleration >= -0.1704097266194472
 Relaxed_CLF_constraint: 0.0259172916412354 x1_input_acceleration
   - 0.1035460233688354 x2_input_acceleration
   + 0.0264250040054321 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2794297337532043 x1_input_acceleration
   + 0.066788911819458 x2_input_acceleration
   + 0.3304388523101807 x3_input_acceleration >= -0.0173078924908709
 CBF_Constraint_sphere_low: 0.1794299483299255 x1_input_acceleration
   - 0.3332107067108154 x2_input_acceleration
   + 0.5304386615753174 x3_input_acceleration >= -0.0755224137795537
 CBF_Constraint_sphere_right: 0.2794299125671387 x1_input_acceleration
   - 0.783210277557373 x2_input_acceleration
   + 0.3304388523101807 x3_input_acceleration >= -0.1695473125062114
 Relaxed_CLF_constraint: 0.0257152915000916 x1_input_acceleration
   - 0.1026049852371216 x2_input_acceleration
   + 0.0262193083763123 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2790257334709167 x1_input_acceleration
   + 0.0686709880828857 x2_input_acceleration
   + 0.3300274610519409 x3_input_acceleration >= -0.0172472973979133
 CBF_Constraint_sphere_low: 0.1790259480476379 x1_input_acceleration
   - 0.3313286304473877 x2_input_acceleration
   + 0.5300272703170776 x3_input_acceleration >= -0.0750644646770674
 CBF_Constraint_sphere_right: 0.2790259122848511 x1_input_acceleration
   - 0.7813282012939453 x2_input_acceleration
   + 0.3300274610519409 x3_input_acceleration >= -0.1686868357280039
 Relaxed_CLF_constraint: 0.0255132913589478 x1_input_acceleration
   - 0.1016639471054077 x2_input_acceleration
   + 0.0260136127471924 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2786217331886292 x1_input_acceleration
   + 0.0705530643463135 x2_input_acceleration
   + 0.3296160697937012 x3_input_acceleration >= -0.0171886396399842
 CBF_Constraint_sphere_low: 0.1786219477653503 x1_input_acceleration
   - 0.32944655418396 x2_input_acceleration
   + 0.5296158790588379 x3_input_acceleration >= -0.0746084529096095
 CBF_Constraint_sphere_right: 0.2786219120025635 x1_input_acceleration
   - 0.7794461250305176 x2_input_acceleration
   + 0.3296160697937012 x3_input_acceleration >= -0.1678282962848249
 Relaxed_CLF_constraint: 0.025311291217804 x1_input_acceleration
   - 0.1007229089736938 x2_input_acceleration
   + 0.0258079171180725 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2782177329063416 x1_input_acceleration
   + 0.0724351406097412 x2_input_acceleration
   + 0.3292046785354614 x3_input_acceleration >= -0.0171319192170835
 CBF_Constraint_sphere_low: 0.1782179474830627 x1_input_acceleration
   - 0.3275644779205322 x2_input_acceleration
   + 0.5292044878005981 x3_input_acceleration >= -0.07415437847718
 CBF_Constraint_sphere_right: 0.2782179117202759 x1_input_acceleration
   - 0.7775640487670898 x2_input_acceleration
   + 0.3292046785354614 x3_input_acceleration >= -0.1669716941766743
 Relaxed_CLF_constraint: 0.0251092910766602 x1_input_acceleration
   - 0.09978187084198 x2_input_acceleration
   + 0.0256022214889526 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.277813732624054 x1_input_acceleration
   + 0.0743172168731689 x2_input_acceleration
   + 0.3287932872772217 x3_input_acceleration >= -0.0170771361292112
 CBF_Constraint_sphere_low: 0.1778139472007751 x1_input_acceleration
   - 0.3256824016571045 x2_input_acceleration
   + 0.5287930965423584 x3_input_acceleration >= -0.0737022413797789
 CBF_Constraint_sphere_right: 0.2778139114379883 x1_input_acceleration
   - 0.7756819725036621 x2_input_acceleration
   + 0.3287932872772217 x3_input_acceleration >= -0.166117029403552
 Relaxed_CLF_constraint: 0.0249072909355164 x1_input_acceleration
   - 0.0988408327102661 x2_input_acceleration
   + 0.0253965258598328 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2774097323417664 x1_input_acceleration
   + 0.0761992931365967 x2_input_acceleration
   + 0.3283820152282715 x3_input_acceleration >= -0.017024309949457
 CBF_Constraint_sphere_low: 0.1774099469184875 x1_input_acceleration
   - 0.3238003253936768 x2_input_acceleration
   + 0.5283818244934082 x3_input_acceleration >= -0.0732520731114136
 CBF_Constraint_sphere_right: 0.2774099111557007 x1_input_acceleration
   - 0.7737998962402344 x2_input_acceleration
   + 0.3283820152282715 x3_input_acceleration >= -0.165264321538548
 Relaxed_CLF_constraint: 0.0247052907943726 x1_input_acceleration
   - 0.0978997945785522 x2_input_acceleration
   + 0.0251908898353577 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2770058512687683 x1_input_acceleration
   + 0.0780813694000244 x2_input_acceleration
   + 0.3279706239700317 x3_input_acceleration >= -0.0169734180179525
 CBF_Constraint_sphere_low: 0.1770060658454895 x1_input_acceleration
   - 0.321918249130249 x2_input_acceleration
   + 0.5279704332351685 x3_input_acceleration >= -0.0728038212099286
 CBF_Constraint_sphere_right: 0.2770060300827026 x1_input_acceleration
   - 0.7719178199768066 x2_input_acceleration
   + 0.3279706239700317 x3_input_acceleration >= -0.1644135479218042
 Relaxed_CLF_constraint: 0.0245033502578735 x1_input_acceleration
   - 0.0969587564468384 x2_input_acceleration
   + 0.0249851942062378 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2766018509864807 x1_input_acceleration
   + 0.0799634456634521 x2_input_acceleration
   + 0.327559232711792 x3_input_acceleration >= -0.0169244468865643
 CBF_Constraint_sphere_low: 0.1766020655632019 x1_input_acceleration
   - 0.3200361728668213 x2_input_acceleration
   + 0.5275590419769287 x3_input_acceleration >= -0.0723574960690116
 CBF_Constraint_sphere_right: 0.276602029800415 x1_input_acceleration
   - 0.7700357437133789 x2_input_acceleration
   + 0.327559232711792 x3_input_acceleration >= -0.1635646951051661
 Relaxed_CLF_constraint: 0.0243013501167297 x1_input_acceleration
   - 0.0960177183151245 x2_input_acceleration
   + 0.0247794985771179 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2761979699134827 x1_input_acceleration
   + 0.0818455219268799 x2_input_acceleration
   + 0.3271478414535522 x3_input_acceleration >= -0.0168774295528828
 CBF_Constraint_sphere_low: 0.1761981844902039 x1_input_acceleration
   - 0.3181540966033936 x2_input_acceleration
   + 0.527147650718689 x3_input_acceleration >= -0.0719131187653496
 CBF_Constraint_sphere_right: 0.276198148727417 x1_input_acceleration
   - 0.7681536674499512 x2_input_acceleration
   + 0.3271478414535522 x3_input_acceleration >= -0.1627177960862453
 Relaxed_CLF_constraint: 0.0240994095802307 x1_input_acceleration
   - 0.0950766801834106 x2_input_acceleration
   + 0.024573802947998 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2757940888404846 x1_input_acceleration
   + 0.0837275981903076 x2_input_acceleration
   + 0.3267364501953125 x3_input_acceleration >= -0.0168323495060762
 CBF_Constraint_sphere_low: 0.1757943034172058 x1_input_acceleration
   - 0.3162720203399658 x2_input_acceleration
   + 0.5267362594604492 x3_input_acceleration >= -0.0714706787485625
 CBF_Constraint_sphere_right: 0.2757942676544189 x1_input_acceleration
   - 0.7662715911865234 x2_input_acceleration
   + 0.3267364501953125 x3_input_acceleration >= -0.1618728343541994
 Relaxed_CLF_constraint: 0.0238974690437317 x1_input_acceleration
   - 0.0941356420516968 x2_input_acceleration
   + 0.0243681073188782 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.275390088558197 x1_input_acceleration
   + 0.0856096744537354 x2_input_acceleration
   + 0.3263250589370728 x3_input_acceleration >= -0.0167891903316125
 CBF_Constraint_sphere_low: 0.1753903031349182 x1_input_acceleration
   - 0.3143899440765381 x2_input_acceleration
   + 0.5263248682022095 x3_input_acceleration >= -0.0710301655645701
 CBF_Constraint_sphere_right: 0.2753902673721313 x1_input_acceleration
   - 0.7643895149230957 x2_input_acceleration
   + 0.3263250589370728 x3_input_acceleration >= -0.1610297934944859
 Relaxed_CLF_constraint: 0.0236954689025879 x1_input_acceleration
   - 0.0931946039199829 x2_input_acceleration
   + 0.0241624116897583 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2749860882759094 x1_input_acceleration
   + 0.0874917507171631 x2_input_acceleration
   + 0.325913667678833 x3_input_acceleration >= -0.0167479684921773
 CBF_Constraint_sphere_low: 0.1749863028526306 x1_input_acceleration
   - 0.3125078678131104 x2_input_acceleration
   + 0.5259134769439697 x3_input_acceleration >= -0.0705915897156061
 CBF_Constraint_sphere_right: 0.2749862670898438 x1_input_acceleration
   - 0.762507438659668 x2_input_acceleration
   + 0.325913667678833 x3_input_acceleration >= -0.1601886899698008
 Relaxed_CLF_constraint: 0.0234934687614441 x1_input_acceleration
   - 0.092253565788269 x2_input_acceleration
   + 0.0239567160606384 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2745820879936218 x1_input_acceleration
   + 0.0893738269805908 x2_input_acceleration
   + 0.3255022764205933 x3_input_acceleration >= -0.0167086839877705
 CBF_Constraint_sphere_low: 0.174582302570343 x1_input_acceleration
   - 0.3106257915496826 x2_input_acceleration
   + 0.52550208568573 x3_input_acceleration >= -0.0701549512016705
 CBF_Constraint_sphere_right: 0.2745822668075562 x1_input_acceleration
   - 0.7606253623962402 x2_input_acceleration
   + 0.3255022764205933 x3_input_acceleration >= -0.159349523780144
 Relaxed_CLF_constraint: 0.0232914686203003 x1_input_acceleration
   - 0.0913125276565552 x2_input_acceleration
   + 0.0237510204315186 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2741782069206238 x1_input_acceleration
   + 0.0912559032440186 x2_input_acceleration
   + 0.3250908851623535 x3_input_acceleration >= -0.0166713531606832
 CBF_Constraint_sphere_low: 0.174178421497345 x1_input_acceleration
   - 0.3087437152862549 x2_input_acceleration
   + 0.5250906944274902 x3_input_acceleration >= -0.0697202604046027
 CBF_Constraint_sphere_right: 0.2741783857345581 x1_input_acceleration
   - 0.7587432861328125 x2_input_acceleration
   + 0.3250908851623535 x3_input_acceleration >= -0.1585123112678174
 Relaxed_CLF_constraint: 0.0230895280838013 x1_input_acceleration
   - 0.0903714895248413 x2_input_acceleration
   + 0.0235453248023987 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2737742066383362 x1_input_acceleration
   + 0.0931379795074463 x2_input_acceleration
   + 0.3246794939041138 x3_input_acceleration >= -0.0166359433022528
 CBF_Constraint_sphere_low: 0.1737744212150574 x1_input_acceleration
   - 0.3068616390228271 x2_input_acceleration
   + 0.5246793031692505 x3_input_acceleration >= -0.0692874965366436
 CBF_Constraint_sphere_right: 0.2737743854522705 x1_input_acceleration
   - 0.7568612098693848 x2_input_acceleration
   + 0.3246794939041138 x3_input_acceleration >= -0.1576770197241372
 Relaxed_CLF_constraint: 0.0228875279426575 x1_input_acceleration
   - 0.0894304513931274 x2_input_acceleration
   + 0.0233396291732788 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2733702063560486 x1_input_acceleration
   + 0.095020055770874 x2_input_acceleration
   + 0.324268102645874 x3_input_acceleration >= -0.0166024707788509
 CBF_Constraint_sphere_low: 0.1733704209327698 x1_input_acceleration
   - 0.3049795627593994 x2_input_acceleration
   + 0.5242679119110107 x3_input_acceleration >= -0.0688566700037129
 CBF_Constraint_sphere_right: 0.2733703851699829 x1_input_acceleration
   - 0.754979133605957 x2_input_acceleration
   + 0.324268102645874 x3_input_acceleration >= -0.1568436655154854
 Relaxed_CLF_constraint: 0.0226855278015137 x1_input_acceleration
   - 0.0884894132614136 x2_input_acceleration
   + 0.0231339335441589 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2729644179344177 x1_input_acceleration
   + 0.0969011783599854 x2_input_acceleration
   + 0.3238548040390015 x3_input_acceleration >= -0.0165703364812423
 CBF_Constraint_sphere_low: 0.1729646325111389 x1_input_acceleration
   - 0.3030984401702881 x2_input_acceleration
   + 0.5238546133041382 x3_input_acceleration >= -0.0684272711033508
 CBF_Constraint_sphere_right: 0.2729645967483521 x1_input_acceleration
   - 0.7530980110168457 x2_input_acceleration
   + 0.3238548040390015 x3_input_acceleration >= -0.1560120548436649
 Relaxed_CLF_constraint: 0.0224826335906982 x1_input_acceleration
   - 0.0875488519668579 x2_input_acceleration
   + 0.0229272842407227 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2725550532341003 x1_input_acceleration
   + 0.0987803936004639 x2_input_acceleration
   + 0.3234380483627319 x3_input_acceleration >= -0.0165389985829997
 CBF_Constraint_sphere_low: 0.1725552678108215 x1_input_acceleration
   - 0.3012192249298096 x2_input_acceleration
   + 0.5234378576278687 x3_input_acceleration >= -0.0679988831786581
 CBF_Constraint_sphere_right: 0.2725552320480347 x1_input_acceleration
   - 0.7512187957763672 x2_input_acceleration
   + 0.3234380483627319 x3_input_acceleration >= -0.1551820511932866
 Relaxed_CLF_constraint: 0.0222779512405396 x1_input_acceleration
   - 0.0866092443466187 x2_input_acceleration
   + 0.0227189064025879 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2721367478370667 x1_input_acceleration
   + 0.1006550788879395 x2_input_acceleration
   + 0.3230127096176147 x3_input_acceleration >= -0.0165067662509465
 CBF_Constraint_sphere_low: 0.1721369624137878 x1_input_acceleration
   - 0.299344539642334 x2_input_acceleration
   + 0.5230125188827515 x3_input_acceleration >= -0.0675700955377018
 CBF_Constraint_sphere_right: 0.272136926651001 x1_input_acceleration
   - 0.7493441104888916 x2_input_acceleration
   + 0.3230127096176147 x3_input_acceleration >= -0.1543530783364885
 Relaxed_CLF_constraint: 0.0220687985420227 x1_input_acceleration
   - 0.0856719017028809 x2_input_acceleration
   + 0.0225062370300293 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2717180848121643 x1_input_acceleration
   + 0.102529764175415 x2_input_acceleration
   + 0.3225877285003662 x3_input_acceleration >= -0.0164764781837207
 CBF_Constraint_sphere_low: 0.1717182993888855 x1_input_acceleration
   - 0.2974698543548584 x2_input_acceleration
   + 0.5225875377655029 x3_input_acceleration >= -0.0671433058056806
 CBF_Constraint_sphere_right: 0.2717182636260986 x1_input_acceleration
   - 0.747469425201416 x2_input_acceleration
   + 0.3225877285003662 x3_input_acceleration >= -0.1535260497444858
 Relaxed_CLF_constraint: 0.0218594670295715 x1_input_acceleration
   - 0.0847345590591431 x2_input_acceleration
   + 0.022293746471405 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2713000178337097 x1_input_acceleration
   + 0.1044049263000488 x2_input_acceleration
   + 0.3221642971038818 x3_input_acceleration >= -0.016448480660182
 CBF_Constraint_sphere_low: 0.1713002324104309 x1_input_acceleration
   - 0.2955946922302246 x2_input_acceleration
   + 0.5221641063690186 x3_input_acceleration >= -0.0667188364196761
 CBF_Constraint_sphere_right: 0.271300196647644 x1_input_acceleration
   - 0.7455942630767822 x2_input_acceleration
   + 0.3221642971038818 x3_input_acceleration >= -0.1527011090406246
 Relaxed_CLF_constraint: 0.0216504335403442 x1_input_acceleration
   - 0.0837969779968262 x2_input_acceleration
   + 0.0220820307731628 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2708807587623596 x1_input_acceleration
   + 0.1062796115875244 x2_input_acceleration
   + 0.3217399120330811 x3_input_acceleration >= -0.0164220780754804
 CBF_Constraint_sphere_low: 0.1708809733390808 x1_input_acceleration
   - 0.293720006942749 x2_input_acceleration
   + 0.5217397212982178 x3_input_acceleration >= -0.0662960215770255
 CBF_Constraint_sphere_right: 0.2708809375762939 x1_input_acceleration
   - 0.7437195777893066 x2_input_acceleration
   + 0.3217399120330811 x3_input_acceleration >= -0.1518779659310928
 Relaxed_CLF_constraint: 0.0214408040046692 x1_input_acceleration
   - 0.0828596353530884 x2_input_acceleration
   + 0.0218698382377625 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2704614996910095 x1_input_acceleration
   + 0.108154296875 x2_input_acceleration
   + 0.3213154077529907 x3_input_acceleration >= -0.0163975915017766
 CBF_Constraint_sphere_low: 0.1704617142677307 x1_input_acceleration
   - 0.2918453216552734 x2_input_acceleration
   + 0.5213152170181274 x3_input_acceleration >= -0.0658751108244552
 CBF_Constraint_sphere_right: 0.2704616785049438 x1_input_acceleration
   - 0.7418448925018311 x2_input_acceleration
   + 0.3213154077529907 x3_input_acceleration >= -0.1510567388325589
 Relaxed_CLF_constraint: 0.0212311744689941 x1_input_acceleration
   - 0.0819222927093506 x2_input_acceleration
   + 0.0216575860977173 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2700422406196594 x1_input_acceleration
   + 0.1100289821624756 x2_input_acceleration
   + 0.3208910226821899 x3_input_acceleration >= -0.0163750592681545
 CBF_Constraint_sphere_low: 0.1700424551963806 x1_input_acceleration
   - 0.2899706363677979 x2_input_acceleration
   + 0.5208908319473267 x3_input_acceleration >= -0.0654561663328842
 CBF_Constraint_sphere_right: 0.2700424194335938 x1_input_acceleration
   - 0.7399702072143555 x2_input_acceleration
   + 0.3208910226821899 x3_input_acceleration >= -0.1502374660741067
 Relaxed_CLF_constraint: 0.0210215449333191 x1_input_acceleration
   - 0.0809849500656128 x2_input_acceleration
   + 0.0214453935623169 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2696228623390198 x1_input_acceleration
   + 0.1119036674499512 x2_input_acceleration
   + 0.3204666376113892 x3_input_acceleration >= -0.0163544461266462
 CBF_Constraint_sphere_low: 0.169623076915741 x1_input_acceleration
   - 0.2880959510803223 x2_input_acceleration
   + 0.5204664468765259 x3_input_acceleration >= -0.0650391468938786
 CBF_Constraint_sphere_right: 0.2696230411529541 x1_input_acceleration
   - 0.7380955219268799 x2_input_acceleration
   + 0.3204666376113892 x3_input_acceleration >= -0.1494201124077576
 Relaxed_CLF_constraint: 0.0208118557929993 x1_input_acceleration
   - 0.080047607421875 x2_input_acceleration
   + 0.0212332010269165 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2692034840583801 x1_input_acceleration
   + 0.1137783527374268 x2_input_acceleration
   + 0.3200423717498779 x3_input_acceleration >= -0.016335787274025
 CBF_Constraint_sphere_low: 0.1692036986351013 x1_input_acceleration
   - 0.2862212657928467 x2_input_acceleration
   + 0.5200421810150146 x3_input_acceleration >= -0.0646240936646777
 CBF_Constraint_sphere_right: 0.2692036628723145 x1_input_acceleration
   - 0.7362208366394043 x2_input_acceleration
   + 0.3200423717498779 x3_input_acceleration >= -0.1486047130302956
 Relaxed_CLF_constraint: 0.0206021666526794 x1_input_acceleration
   - 0.0791102647781372 x2_input_acceleration
   + 0.0210210680961609 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2687839865684509 x1_input_acceleration
   + 0.1156530380249023 x2_input_acceleration
   + 0.3196181058883667 x3_input_acceleration >= -0.0163190475629215
 CBF_Constraint_sphere_low: 0.1687842011451721 x1_input_acceleration
   - 0.2843465805053711 x2_input_acceleration
   + 0.5196179151535034 x3_input_acceleration >= -0.0642109655374462
 CBF_Constraint_sphere_right: 0.2687841653823853 x1_input_acceleration
   - 0.7343461513519287 x2_input_acceleration
   + 0.3196181058883667 x3_input_acceleration >= -0.1477912327943406
 Relaxed_CLF_constraint: 0.0203924179077148 x1_input_acceleration
   - 0.0781729221343994 x2_input_acceleration
   + 0.0208089351654053 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2683644890785217 x1_input_acceleration
   + 0.1175277233123779 x2_input_acceleration
   + 0.3191938400268555 x3_input_acceleration >= -0.0163042430641142
 CBF_Constraint_sphere_low: 0.1683647036552429 x1_input_acceleration
   - 0.2824718952178955 x2_input_acceleration
   + 0.5191936492919922 x3_input_acceleration >= -0.0637997726225109
 CBF_Constraint_sphere_right: 0.2683646678924561 x1_input_acceleration
   - 0.7324714660644531 x2_input_acceleration
   + 0.3191938400268555 x3_input_acceleration >= -0.1469796877706818
 Relaxed_CLF_constraint: 0.0201826691627502 x1_input_acceleration
   - 0.0772355794906616 x2_input_acceleration
   + 0.0205968022346497 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2679449915885925 x1_input_acceleration
   + 0.1194024085998535 x2_input_acceleration
   + 0.3187695741653442 x3_input_acceleration >= -0.0162913737776031
 CBF_Constraint_sphere_low: 0.1679452061653137 x1_input_acceleration
   - 0.2805972099304199 x2_input_acceleration
   + 0.518769383430481 x3_input_acceleration >= -0.0633905149198718
 CBF_Constraint_sphere_right: 0.2679451704025269 x1_input_acceleration
   - 0.7305967807769775 x2_input_acceleration
   + 0.3187695741653442 x3_input_acceleration >= -0.1461700779593193
 Relaxed_CLF_constraint: 0.0199729204177856 x1_input_acceleration
   - 0.0762982368469238 x2_input_acceleration
   + 0.020384669303894 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2675254940986633 x1_input_acceleration
   + 0.1212770938873291 x2_input_acceleration
   + 0.3183454275131226 x3_input_acceleration >= -0.0162804586782507
 CBF_Constraint_sphere_low: 0.1675257086753845 x1_input_acceleration
   - 0.2787225246429443 x2_input_acceleration
   + 0.5183452367782593 x3_input_acceleration >= -0.062983223325309
 CBF_Constraint_sphere_right: 0.2675256729125977 x1_input_acceleration
   - 0.728722095489502 x2_input_acceleration
   + 0.3183454275131226 x3_input_acceleration >= -0.1453624223351155
 Relaxed_CLF_constraint: 0.019763171672821 x1_input_acceleration
   - 0.075360894203186 x2_input_acceleration
   + 0.0201725959777832 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2671061158180237 x1_input_acceleration
   + 0.1231517791748047 x2_input_acceleration
   + 0.3179211616516113 x3_input_acceleration >= -0.0162714757118053
 CBF_Constraint_sphere_low: 0.1671063303947449 x1_input_acceleration
   - 0.2768478393554688 x2_input_acceleration
   + 0.517920970916748 x3_input_acceleration >= -0.062577845982284
 CBF_Constraint_sphere_right: 0.267106294631958 x1_input_acceleration
   - 0.7268474102020264 x2_input_acceleration
   + 0.3179211616516113 x3_input_acceleration >= -0.1445566988438293
 Relaxed_CLF_constraint: 0.0195534825325012 x1_input_acceleration
   - 0.0744235515594482 x2_input_acceleration
   + 0.0199604630470276 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2666866183280945 x1_input_acceleration
   + 0.1250264644622803 x2_input_acceleration
   + 0.3174968957901001 x3_input_acceleration >= -0.0162644120118906
 CBF_Constraint_sphere_low: 0.1666868329048157 x1_input_acceleration
   - 0.2749731540679932 x2_input_acceleration
   + 0.5174967050552368 x3_input_acceleration >= -0.0621743938662413
 CBF_Constraint_sphere_right: 0.2666867971420288 x1_input_acceleration
   - 0.7249727249145508 x2_input_acceleration
   + 0.3174968957901001 x3_input_acceleration >= -0.1437528946190631
 Relaxed_CLF_constraint: 0.0193437337875366 x1_input_acceleration
   - 0.0734862089157104 x2_input_acceleration
   + 0.019748330116272 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2662671208381653 x1_input_acceleration
   + 0.1269011497497559 x2_input_acceleration
   + 0.3170726299285889 x3_input_acceleration >= -0.016259283524272
 CBF_Constraint_sphere_low: 0.1662673354148865 x1_input_acceleration
   - 0.2730984687805176 x2_input_acceleration
   + 0.5170724391937256 x3_input_acceleration >= -0.0617728769624947
 CBF_Constraint_sphere_right: 0.2662672996520996 x1_input_acceleration
   - 0.7230980396270752 x2_input_acceleration
   + 0.3170726299285889 x3_input_acceleration >= -0.142951025606593
 Relaxed_CLF_constraint: 0.019133985042572 x1_input_acceleration
   - 0.0725488662719727 x2_input_acceleration
   + 0.0195361971855164 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2658476233482361 x1_input_acceleration
   + 0.1287758350372314 x2_input_acceleration
   + 0.3166483640670776 x3_input_acceleration >= -0.0162560902489496
 CBF_Constraint_sphere_low: 0.1658478379249573 x1_input_acceleration
   - 0.271223783493042 x2_input_acceleration
   + 0.5166481733322144 x3_input_acceleration >= -0.0613732952710443
 CBF_Constraint_sphere_right: 0.2658478021621704 x1_input_acceleration
   - 0.7212233543395996 x2_input_acceleration
   + 0.3166483640670776 x3_input_acceleration >= -0.1421510918064192
 Relaxed_CLF_constraint: 0.0189242362976074 x1_input_acceleration
   - 0.0716115236282349 x2_input_acceleration
   + 0.0193240642547607 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2654281258583069 x1_input_acceleration
   + 0.130650520324707 x2_input_acceleration
   + 0.3162240982055664 x3_input_acceleration >= -0.0162548321859234
 CBF_Constraint_sphere_low: 0.1654283404350281 x1_input_acceleration
   - 0.2693490982055664 x2_input_acceleration
   + 0.5162239074707031 x3_input_acceleration >= -0.0609756487918901
 CBF_Constraint_sphere_right: 0.2654283046722412 x1_input_acceleration
   - 0.719348669052124 x2_input_acceleration
   + 0.3162240982055664 x3_input_acceleration >= -0.1413530932185415
 Relaxed_CLF_constraint: 0.0187144875526428 x1_input_acceleration
   - 0.0706741809844971 x2_input_acceleration
   + 0.0191119313240051 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2650085091590881 x1_input_acceleration
   + 0.1325249671936035 x2_input_acceleration
   + 0.3157997131347656 x3_input_acceleration >= -0.0162554589180971
 CBF_Constraint_sphere_low: 0.1650087237358093 x1_input_acceleration
   - 0.2674746513366699 x2_input_acceleration
   + 0.5157995223999023 x3_input_acceleration >= -0.0605799288311403
 CBF_Constraint_sphere_right: 0.2650086879730225 x1_input_acceleration
   - 0.7174742221832275 x2_input_acceleration
   + 0.3157997131347656 x3_input_acceleration >= -0.1405570807536526
 Relaxed_CLF_constraint: 0.0185046792030334 x1_input_acceleration
   - 0.0697369575500488 x2_input_acceleration
   + 0.0188997387886047 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2645888924598694 x1_input_acceleration
   + 0.1343994140625 x2_input_acceleration
   + 0.3153753280639648 x3_input_acceleration >= -0.0162580205162343
 CBF_Constraint_sphere_low: 0.1645891070365906 x1_input_acceleration
   - 0.2656002044677734 x2_input_acceleration
   + 0.5153751373291016 x3_input_acceleration >= -0.0601861437363539
 CBF_Constraint_sphere_right: 0.2645890712738037 x1_input_acceleration
   - 0.7155997753143311 x2_input_acceleration
   + 0.3153753280639648 x3_input_acceleration >= -0.1397630031547271
 Relaxed_CLF_constraint: 0.0182948708534241 x1_input_acceleration
   - 0.0687997341156006 x2_input_acceleration
   + 0.0186875462532043 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2641693949699402 x1_input_acceleration
   + 0.1362740993499756 x2_input_acceleration
   + 0.3149510622024536 x3_input_acceleration >= -0.0162625677437213
 CBF_Constraint_sphere_low: 0.1641696095466614 x1_input_acceleration
   - 0.2637255191802979 x2_input_acceleration
   + 0.5149508714675903 x3_input_acceleration >= -0.0597943025477129
 CBF_Constraint_sphere_right: 0.2641695737838745 x1_input_acceleration
   - 0.7137250900268555 x2_input_acceleration
   + 0.3149510622024536 x3_input_acceleration >= -0.1389708098573627
 Relaxed_CLF_constraint: 0.0180851221084595 x1_input_acceleration
   - 0.0678623914718628 x2_input_acceleration
   + 0.0184754133224487 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2637423872947693 x1_input_acceleration
   + 0.1381444931030273 x2_input_acceleration
   + 0.3145185708999634 x3_input_acceleration >= -0.0162664698178979
 CBF_Constraint_sphere_low: 0.1637426018714905 x1_input_acceleration
   - 0.2618551254272461 x2_input_acceleration
   + 0.5145183801651001 x3_input_acceleration >= -0.0594022274769707
 CBF_Constraint_sphere_right: 0.2637425661087036 x1_input_acceleration
   - 0.7118546962738037 x2_input_acceleration
   + 0.3145185708999634 x3_input_acceleration >= -0.1381797953064071
 Relaxed_CLF_constraint: 0.017871618270874 x1_input_acceleration
   - 0.0669271945953369 x2_input_acceleration
   + 0.0182591676712036 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2633113265037537 x1_input_acceleration
   + 0.1400129795074463 x2_input_acceleration
   + 0.3140830993652344 x3_input_acceleration >= -0.0162711706006477
 CBF_Constraint_sphere_low: 0.1633115410804749 x1_input_acceleration
   - 0.2599866390228271 x2_input_acceleration
   + 0.5140829086303711 x3_input_acceleration >= -0.059011237216582
 CBF_Constraint_sphere_right: 0.263311505317688 x1_input_acceleration
   - 0.7099862098693848 x2_input_acceleration
   + 0.3140830993652344 x3_input_acceleration >= -0.1373903900860582
 Relaxed_CLF_constraint: 0.0176560878753662 x1_input_acceleration
   - 0.0659929513931274 x2_input_acceleration
   + 0.0180414319038391 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2628797888755798 x1_input_acceleration
   + 0.1418814659118652 x2_input_acceleration
   + 0.313647985458374 x3_input_acceleration >= -0.0162777981376664
 CBF_Constraint_sphere_low: 0.162880003452301 x1_input_acceleration
   - 0.2581181526184082 x2_input_acceleration
   + 0.5136477947235107 x3_input_acceleration >= -0.0586222333150218
 CBF_Constraint_sphere_right: 0.2628799676895142 x1_input_acceleration
   - 0.7081177234649658 x2_input_acceleration
   + 0.313647985458374 x3_input_acceleration >= -0.1366029116199356
 Relaxed_CLF_constraint: 0.0174403190612793 x1_input_acceleration
   - 0.065058708190918 x2_input_acceleration
   + 0.0178238749504089 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2624480128288269 x1_input_acceleration
   + 0.1437499523162842 x2_input_acceleration
   + 0.3132131099700928 x3_input_acceleration >= -0.0162863651214679
 CBF_Constraint_sphere_low: 0.1624482274055481 x1_input_acceleration
   - 0.2562496662139893 x2_input_acceleration
   + 0.5132129192352295 x3_input_acceleration >= -0.0582352046229829
 CBF_Constraint_sphere_right: 0.2624481916427612 x1_input_acceleration
   - 0.7062492370605469 x2_input_acceleration
   + 0.3132131099700928 x3_input_acceleration >= -0.1358173726005744
 Relaxed_CLF_constraint: 0.0172244310379028 x1_input_acceleration
   - 0.0641244649887085 x2_input_acceleration
   + 0.0176064372062683 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2620159983634949 x1_input_acceleration
   + 0.1456184387207031 x2_input_acceleration
   + 0.3127784729003906 x3_input_acceleration >= -0.0162968715509436
 CBF_Constraint_sphere_low: 0.1620162129402161 x1_input_acceleration
   - 0.2543811798095703 x2_input_acceleration
   + 0.5127782821655273 x3_input_acceleration >= -0.0578501511393569
 CBF_Constraint_sphere_right: 0.2620161771774292 x1_input_acceleration
   - 0.7043807506561279 x2_input_acceleration
   + 0.3127784729003906 x3_input_acceleration >= -0.1350337730268662
 Relaxed_CLF_constraint: 0.0170084238052368 x1_input_acceleration
   - 0.063190221786499 x2_input_acceleration
   + 0.0173891186714172 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2615839838981628 x1_input_acceleration
   + 0.1474869251251221 x2_input_acceleration
   + 0.3123438358306885 x3_input_acceleration >= -0.0163093113740814
 CBF_Constraint_sphere_low: 0.161584198474884 x1_input_acceleration
   - 0.2525126934051514 x2_input_acceleration
   + 0.5123436450958252 x3_input_acceleration >= -0.0574670310493929
 CBF_Constraint_sphere_right: 0.2615841627120972 x1_input_acceleration
   - 0.702512264251709 x2_input_acceleration
   + 0.3123438358306885 x3_input_acceleration >= -0.13425210684682
 Relaxed_CLF_constraint: 0.0167924165725708 x1_input_acceleration
   - 0.0622559785842896 x2_input_acceleration
   + 0.0171718001365662 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2611518502235413 x1_input_acceleration
   + 0.149355411529541 x2_input_acceleration
   + 0.3119094371795654 x3_input_acceleration >= -0.0163237062075027
 CBF_Constraint_sphere_low: 0.1611520648002625 x1_input_acceleration
   - 0.2506442070007324 x2_input_acceleration
   + 0.5119092464447021 x3_input_acceleration >= -0.0570858957719992
 CBF_Constraint_sphere_right: 0.2611520290374756 x1_input_acceleration
   - 0.70064377784729 x2_input_acceleration
   + 0.3119094371795654 x3_input_acceleration >= -0.1334723956770466
 Relaxed_CLF_constraint: 0.01657634973526 x1_input_acceleration
   - 0.0613217353820801 x2_input_acceleration
   + 0.0169546008110046 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2607197165489197 x1_input_acceleration
   + 0.15122389793396 x2_input_acceleration
   + 0.3114749193191528 x3_input_acceleration >= -0.0163400158171406
 CBF_Constraint_sphere_low: 0.1607199311256409 x1_input_acceleration
   - 0.2487757205963135 x2_input_acceleration
   + 0.5114747285842896 x3_input_acceleration >= -0.0567066633499046
 CBF_Constraint_sphere_right: 0.260719895362854 x1_input_acceleration
   - 0.6987752914428711 x2_input_acceleration
   + 0.3114749193191528 x3_input_acceleration >= -0.13269459928349
 Relaxed_CLF_constraint: 0.0163602828979492 x1_input_acceleration
   - 0.0603874921798706 x2_input_acceleration
   + 0.0167373418807983 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2602874636650085 x1_input_acceleration
   + 0.1530923843383789 x2_input_acceleration
   + 0.3110405206680298 x3_input_acceleration >= -0.016358261845253
 CBF_Constraint_sphere_low: 0.1602876782417297 x1_input_acceleration
   - 0.2469072341918945 x2_input_acceleration
   + 0.5110403299331665 x3_input_acceleration >= -0.0563293852276538
 CBF_Constraint_sphere_right: 0.2602876424789429 x1_input_acceleration
   - 0.6969068050384521 x2_input_acceleration
   + 0.3110405206680298 x3_input_acceleration >= -0.1319187393083971
 Relaxed_CLF_constraint: 0.0161441564559937 x1_input_acceleration
   - 0.0594532489776611 x2_input_acceleration
   + 0.0165201425552368 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2598550915718079 x1_input_acceleration
   + 0.1549608707427979 x2_input_acceleration
   + 0.3106067180633545 x3_input_acceleration >= -0.0163785183458117
 CBF_Constraint_sphere_low: 0.1598553061485291 x1_input_acceleration
   - 0.2450387477874756 x2_input_acceleration
   + 0.5106065273284912 x3_input_acceleration >= -0.0559541831428888
 CBF_Constraint_sphere_right: 0.2598552703857422 x1_input_acceleration
   - 0.6950383186340332 x2_input_acceleration
   + 0.3106067180633545 x3_input_acceleration >= -0.1311448898057398
 Relaxed_CLF_constraint: 0.0159279704093933 x1_input_acceleration
   - 0.0585190057754517 x2_input_acceleration
   + 0.0163032412528992 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2594228386878967 x1_input_acceleration
   + 0.1568293571472168 x2_input_acceleration
   + 0.3101723194122314 x3_input_acceleration >= -0.0164006310564148
 CBF_Constraint_sphere_low: 0.1594230532646179 x1_input_acceleration
   - 0.2431702613830566 x2_input_acceleration
   + 0.5101721286773682 x3_input_acceleration >= -0.0555807717031286
 CBF_Constraint_sphere_right: 0.2594230175018311 x1_input_acceleration
   - 0.6931698322296143 x2_input_acceleration
   + 0.3101723194122314 x3_input_acceleration >= -0.1303728965131376
 Relaxed_CLF_constraint: 0.0157118439674377 x1_input_acceleration
   - 0.0575847625732422 x2_input_acceleration
   + 0.0160860419273376 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2589905858039856 x1_input_acceleration
   + 0.1586978435516357 x2_input_acceleration
   + 0.3097379207611084 x3_input_acceleration >= -0.0164246771601114
 CBF_Constraint_sphere_low: 0.1589908003807068 x1_input_acceleration
   - 0.2413017749786377 x2_input_acceleration
   + 0.5097377300262451 x3_input_acceleration >= -0.055209293656462
 CBF_Constraint_sphere_right: 0.2589907646179199 x1_input_acceleration
   - 0.6913013458251953 x2_input_acceleration
   + 0.3097379207611084 x3_input_acceleration >= -0.129602836613629
 Relaxed_CLF_constraint: 0.0154957175254822 x1_input_acceleration
   - 0.0566505193710327 x2_input_acceleration
   + 0.0158688426017761 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2585583329200745 x1_input_acceleration
   + 0.1605660915374756 x2_input_acceleration
   + 0.3093029260635376 x3_input_acceleration >= -0.0164505453363738
 CBF_Constraint_sphere_low: 0.1585585474967957 x1_input_acceleration
   - 0.2394335269927979 x2_input_acceleration
   + 0.5093027353286743 x3_input_acceleration >= -0.0548396257614435
 CBF_Constraint_sphere_right: 0.2585585117340088 x1_input_acceleration
   - 0.6894330978393555 x2_input_acceleration
   + 0.3093029260635376 x3_input_acceleration >= -0.1288347001144855
 Relaxed_CLF_constraint: 0.0152795910835266 x1_input_acceleration
   - 0.0557163953781128 x2_input_acceleration
   + 0.0156513452529907 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2581263184547424 x1_input_acceleration
   + 0.1624345779418945 x2_input_acceleration
   + 0.3088682889938354 x3_input_acceleration >= -0.016478452084035
 CBF_Constraint_sphere_low: 0.1581265330314636 x1_input_acceleration
   - 0.2375650405883789 x2_input_acceleration
   + 0.5088680982589722 x3_input_acceleration >= -0.0544719725960029
 CBF_Constraint_sphere_right: 0.2581264972686768 x1_input_acceleration
   - 0.6875646114349365 x2_input_acceleration
   + 0.3088682889938354 x3_input_acceleration >= -0.1280685008589627
 Relaxed_CLF_constraint: 0.0150635838508606 x1_input_acceleration
   - 0.0547821521759033 x2_input_acceleration
   + 0.0154340267181396 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2576906085014343 x1_input_acceleration
   + 0.1643009185791016 x2_input_acceleration
   + 0.3084297180175781 x3_input_acceleration >= -0.0165070331280447
 CBF_Constraint_sphere_low: 0.1576908230781555 x1_input_acceleration
   - 0.2356986999511719 x2_input_acceleration
   + 0.5084295272827148 x3_input_acceleration >= -0.0541052142636659
 CBF_Constraint_sphere_right: 0.2576907873153687 x1_input_acceleration
   - 0.6856982707977295 x2_input_acceleration
   + 0.3084297180175781 x3_input_acceleration >= -0.1273038878496534
 Relaxed_CLF_constraint: 0.0148457288742065 x1_input_acceleration
   - 0.0538489818572998 x2_input_acceleration
   + 0.015214741230011 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2572466731071472 x1_input_acceleration
   + 0.1661622524261475 x2_input_acceleration
   + 0.3079816102981567 x3_input_acceleration >= -0.016534604309468
 CBF_Constraint_sphere_low: 0.1572468876838684 x1_input_acceleration
   - 0.233837366104126 x2_input_acceleration
   + 0.5079814195632935 x3_input_acceleration >= -0.0537379050235792
 CBF_Constraint_sphere_right: 0.2572468519210815 x1_input_acceleration
   - 0.6838369369506836 x2_input_acceleration
   + 0.3079816102981567 x3_input_acceleration >= -0.1265403928608115
 Relaxed_CLF_constraint: 0.014623761177063 x1_input_acceleration
   - 0.0529183149337769 x2_input_acceleration
   + 0.0149906873703003 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2567769885063171 x1_input_acceleration
   + 0.1680080890655518 x2_input_acceleration
   + 0.3075040578842163 x3_input_acceleration >= -0.0165549713582728
 CBF_Constraint_sphere_low: 0.1567772030830383 x1_input_acceleration
   - 0.2319915294647217 x2_input_acceleration
   + 0.507503867149353 x3_input_acceleration >= -0.0533648340803673
 CBF_Constraint_sphere_right: 0.2567771673202515 x1_input_acceleration
   - 0.6819911003112793 x2_input_acceleration
   + 0.3075040578842163 x3_input_acceleration >= -0.1257762800440153
 Relaxed_CLF_constraint: 0.0143889188766479 x1_input_acceleration
   - 0.0519953966140747 x2_input_acceleration
   + 0.0147519111633301 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2562922835350037 x1_input_acceleration
   + 0.1698448657989502 x2_input_acceleration
   + 0.3070094585418701 x3_input_acceleration >= -0.0165719551680056
 CBF_Constraint_sphere_low: 0.1562924981117249 x1_input_acceleration
   - 0.2301547527313232 x2_input_acceleration
   + 0.5070092678070068 x3_input_acceleration >= -0.0529892382032543
 CBF_Constraint_sphere_right: 0.256292462348938 x1_input_acceleration
   - 0.6801543235778809 x2_input_acceleration
   + 0.3070094585418701 x3_input_acceleration >= -0.1250126344431847
 Relaxed_CLF_constraint: 0.0141465663909912 x1_input_acceleration
   - 0.0510770082473755 x2_input_acceleration
   + 0.014504611492157 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2558004260063171 x1_input_acceleration
   + 0.1716773509979248 x2_input_acceleration
   + 0.3065067529678345 x3_input_acceleration >= -0.0165883401013908
 CBF_Constraint_sphere_low: 0.1558006405830383 x1_input_acceleration
   - 0.2283222675323486 x2_input_acceleration
   + 0.5065065622329712 x3_input_acceleration >= -0.0526134487605653
 CBF_Constraint_sphere_right: 0.2558006048202515 x1_input_acceleration
   - 0.6783218383789062 x2_input_acceleration
   + 0.3065067529678345 x3_input_acceleration >= -0.1242502138657576
 Relaxed_CLF_constraint: 0.0139006376266479 x1_input_acceleration
   - 0.0501607656478882 x2_input_acceleration
   + 0.0142532587051392 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2553107142448425 x1_input_acceleration
   + 0.1735110282897949 x2_input_acceleration
   + 0.3060065507888794 x3_input_acceleration >= -0.0166074117180252
 CBF_Constraint_sphere_low: 0.1553109288215637 x1_input_acceleration
   - 0.2264885902404785 x2_input_acceleration
   + 0.5060063600540161 x3_input_acceleration >= -0.0522402506339127
 CBF_Constraint_sphere_right: 0.2553108930587769 x1_input_acceleration
   - 0.6764881610870361 x2_input_acceleration
   + 0.3060065507888794 x3_input_acceleration >= -0.1234899733327742
 Relaxed_CLF_constraint: 0.0136557817459106 x1_input_acceleration
   - 0.0492439270019531 x2_input_acceleration
   + 0.0140031576156616 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2548170685768127 x1_input_acceleration
   + 0.175342321395874 x2_input_acceleration
   + 0.3055018186569214 x3_input_acceleration >= -0.0166270073283151
 CBF_Constraint_sphere_low: 0.1548172831535339 x1_input_acceleration
   - 0.2246572971343994 x2_input_acceleration
   + 0.5055016279220581 x3_input_acceleration >= -0.0518677970376566
 CBF_Constraint_sphere_right: 0.2548172473907471 x1_input_acceleration
   - 0.674656867980957 x2_input_acceleration
   + 0.3055018186569214 x3_input_acceleration >= -0.1227312700710895
 Relaxed_CLF_constraint: 0.0134089589118958 x1_input_acceleration
   - 0.0483282804489136 x2_input_acceleration
   + 0.0137507915496826 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2543172240257263 x1_input_acceleration
   + 0.1771700382232666 x2_input_acceleration
   + 0.3049900531768799 x3_input_acceleration >= -0.0166463513542553
 CBF_Constraint_sphere_low: 0.1543174386024475 x1_input_acceleration
   - 0.2228295803070068 x2_input_acceleration
   + 0.5049898624420166 x3_input_acceleration >= -0.051495413721456
 CBF_Constraint_sphere_right: 0.2543174028396606 x1_input_acceleration
   - 0.6728291511535645 x2_input_acceleration
   + 0.3049900531768799 x3_input_acceleration >= -0.1219738351414931
 Relaxed_CLF_constraint: 0.0131590366363525 x1_input_acceleration
   - 0.0474144220352173 x2_input_acceleration
   + 0.0134949088096619 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2538160681724548 x1_input_acceleration
   + 0.1789968013763428 x2_input_acceleration
   + 0.3044767379760742 x3_input_acceleration >= -0.0166671338335093
 CBF_Constraint_sphere_low: 0.153816282749176 x1_input_acceleration
   - 0.2210028171539307 x2_input_acceleration
   + 0.5044765472412109 x3_input_acceleration >= -0.0511245701862905
 CBF_Constraint_sphere_right: 0.2538162469863892 x1_input_acceleration
   - 0.6710023880004883 x2_input_acceleration
   + 0.3044767379760742 x3_input_acceleration >= -0.1212182439762911
 Relaxed_CLF_constraint: 0.0129084587097168 x1_input_acceleration
   - 0.0465010404586792 x2_input_acceleration
   + 0.013238251209259 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2533138394355774 x1_input_acceleration
   + 0.1808228492736816 x2_input_acceleration
   + 0.3039619922637939 x3_input_acceleration >= -0.0166894242025052
 CBF_Constraint_sphere_low: 0.1533140540122986 x1_input_acceleration
   - 0.2191767692565918 x2_input_acceleration
   + 0.5039618015289307 x3_input_acceleration >= -0.0507552881849321
 CBF_Constraint_sphere_right: 0.2533140182495117 x1_input_acceleration
   - 0.6691763401031494 x2_input_acceleration
   + 0.3039619922637939 x3_input_acceleration >= -0.1204644646841336
 Relaxed_CLF_constraint: 0.0126573443412781 x1_input_acceleration
   - 0.0455880165100098 x2_input_acceleration
   + 0.0129808783531189 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2528116106987 x1_input_acceleration
   + 0.1826491355895996 x2_input_acceleration
   + 0.3034476041793823 x3_input_acceleration >= -0.0167137164294765
 CBF_Constraint_sphere_low: 0.1528118252754211 x1_input_acceleration
   - 0.2173504829406738 x2_input_acceleration
   + 0.503447413444519 x3_input_acceleration >= -0.0503879961206316
 CBF_Constraint_sphere_right: 0.2528117895126343 x1_input_acceleration
   - 0.6673500537872314 x2_input_acceleration
   + 0.3034476041793823 x3_input_acceleration >= -0.1197125859221521
 Relaxed_CLF_constraint: 0.0124062299728394 x1_input_acceleration
   - 0.0446748733520508 x2_input_acceleration
   + 0.0127236843109131 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2523093819618225 x1_input_acceleration
   + 0.1844751834869385 x2_input_acceleration
   + 0.3029328584671021 x3_input_acceleration >= -0.0167398585718864
 CBF_Constraint_sphere_low: 0.1523095965385437 x1_input_acceleration
   - 0.215524435043335 x2_input_acceleration
   + 0.5029326677322388 x3_input_acceleration >= -0.0500225658926873
 CBF_Constraint_sphere_right: 0.2523095607757568 x1_input_acceleration
   - 0.6655240058898926 x2_input_acceleration
   + 0.3029328584671021 x3_input_acceleration >= -0.1189626584034086
 Relaxed_CLF_constraint: 0.0121551156044006 x1_input_acceleration
   - 0.0437618494033813 x2_input_acceleration
   + 0.0124663114547729 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2518070340156555 x1_input_acceleration
   + 0.1863012313842773 x2_input_acceleration
   + 0.3024181127548218 x3_input_acceleration >= -0.0167679115293119
 CBF_Constraint_sphere_low: 0.1518072485923767 x1_input_acceleration
   - 0.2136983871459961 x2_input_acceleration
   + 0.5024179220199585 x3_input_acceleration >= -0.0496590524402102
 CBF_Constraint_sphere_right: 0.2518072128295898 x1_input_acceleration
   - 0.6636979579925537 x2_input_acceleration
   + 0.3024181127548218 x3_input_acceleration >= -0.11821464169967
 Relaxed_CLF_constraint: 0.0119039416313171 x1_input_acceleration
   - 0.0428488254547119 x2_input_acceleration
   + 0.0122089385986328 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2513046860694885 x1_input_acceleration
   + 0.1881272792816162 x2_input_acceleration
   + 0.3019033670425415 x3_input_acceleration >= -0.0167978903705027
 CBF_Constraint_sphere_low: 0.1513049006462097 x1_input_acceleration
   - 0.2118723392486572 x2_input_acceleration
   + 0.5019031763076782 x3_input_acceleration >= -0.0492974648714985
 CBF_Constraint_sphere_right: 0.2513048648834229 x1_input_acceleration
   - 0.6618719100952148 x2_input_acceleration
   + 0.3019033670425415 x3_input_acceleration >= -0.1174685508796968
 Relaxed_CLF_constraint: 0.0116527676582336 x1_input_acceleration
   - 0.0419358015060425 x2_input_acceleration
   + 0.0119515657424927 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2508023381233215 x1_input_acceleration
   + 0.1899533271789551 x2_input_acceleration
   + 0.3013886213302612 x3_input_acceleration >= -0.0168297950954589
 CBF_Constraint_sphere_low: 0.1508025527000427 x1_input_acceleration
   - 0.2100462913513184 x2_input_acceleration
   + 0.5013884305953979 x3_input_acceleration >= -0.0489378031865521
 CBF_Constraint_sphere_right: 0.2508025169372559 x1_input_acceleration
   - 0.660045862197876 x2_input_acceleration
   + 0.3013886213302612 x3_input_acceleration >= -0.1167243859434889
 Relaxed_CLF_constraint: 0.0114015936851501 x1_input_acceleration
   - 0.041022777557373 x2_input_acceleration
   + 0.0116941928863525 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2502999901771545 x1_input_acceleration
   + 0.191779613494873 x2_input_acceleration
   + 0.3008742332458496 x3_input_acceleration >= -0.0168637023665511
 CBF_Constraint_sphere_low: 0.1503002047538757 x1_input_acceleration
   - 0.2082200050354004 x2_input_acceleration
   + 0.5008740425109863 x3_input_acceleration >= -0.0485801321268241
 CBF_Constraint_sphere_right: 0.2503001689910889 x1_input_acceleration
   - 0.658219575881958 x2_input_acceleration
   + 0.3008742332458496 x3_input_acceleration >= -0.1159821222256176
 Relaxed_CLF_constraint: 0.0111504197120667 x1_input_acceleration
   - 0.0401096343994141 x2_input_acceleration
   + 0.0114369988441467 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2497977614402771 x1_input_acceleration
   + 0.1936056613922119 x2_input_acceleration
   + 0.3003596067428589 x3_input_acceleration >= -0.0168994917766036
 CBF_Constraint_sphere_low: 0.1497979760169983 x1_input_acceleration
   - 0.2063939571380615 x2_input_acceleration
   + 0.5003594160079956 x3_input_acceleration >= -0.0482243610874399
 CBF_Constraint_sphere_right: 0.2497979402542114 x1_input_acceleration
   - 0.6563935279846191 x2_input_acceleration
   + 0.3003596067428589 x3_input_acceleration >= -0.1152418419745166
 Relaxed_CLF_constraint: 0.0108993053436279 x1_input_acceleration
   - 0.0391966104507446 x2_input_acceleration
   + 0.0111796855926514 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2492954134941101 x1_input_acceleration
   + 0.1954317092895508 x2_input_acceleration
   + 0.2998448610305786 x3_input_acceleration >= -0.0169371742178705
 CBF_Constraint_sphere_low: 0.1492956280708313 x1_input_acceleration
   - 0.2045679092407227 x2_input_acceleration
   + 0.4998446702957153 x3_input_acceleration >= -0.0478704771188042
 CBF_Constraint_sphere_right: 0.2492955923080444 x1_input_acceleration
   - 0.6545674800872803 x2_input_acceleration
   + 0.2998448610305786 x3_input_acceleration >= -0.1145034547546194
 Relaxed_CLF_constraint: 0.0106481313705444 x1_input_acceleration
   - 0.0382835865020752 x2_input_acceleration
   + 0.0109223127365112 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2487935423851013 x1_input_acceleration
   + 0.1972579956054688 x2_input_acceleration
   + 0.2993307113647461 x3_input_acceleration >= -0.0169769545822348
 CBF_Constraint_sphere_low: 0.1487937569618225 x1_input_acceleration
   - 0.2027416229248047 x2_input_acceleration
   + 0.4993305206298828 x3_input_acceleration >= -0.0475186791523769
 CBF_Constraint_sphere_right: 0.2487937211990356 x1_input_acceleration
   - 0.6527411937713623 x2_input_acceleration
   + 0.2993307113647461 x3_input_acceleration >= -0.1137670641300629
 Relaxed_CLF_constraint: 0.01039719581604 x1_input_acceleration
   - 0.0373704433441162 x2_input_acceleration
   + 0.010665237903595 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2482913136482239 x1_input_acceleration
   + 0.1990840435028076 x2_input_acceleration
   + 0.2988159656524658 x3_input_acceleration >= -0.017018503534851
 CBF_Constraint_sphere_low: 0.1482915282249451 x1_input_acceleration
   - 0.2009155750274658 x2_input_acceleration
   + 0.4988157749176025 x3_input_acceleration >= -0.0471686557346388
 CBF_Constraint_sphere_right: 0.2482914924621582 x1_input_acceleration
   - 0.6509151458740234 x2_input_acceleration
   + 0.2988159656524658 x3_input_acceleration >= -0.1130325434215257
 Relaxed_CLF_constraint: 0.0101460814476013 x1_input_acceleration
   - 0.0364574193954468 x2_input_acceleration
   + 0.0104078650474548 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2477889657020569 x1_input_acceleration
   + 0.2009100914001465 x2_input_acceleration
   + 0.2983012199401855 x3_input_acceleration >= -0.0170619635419783
 CBF_Constraint_sphere_low: 0.1477891802787781 x1_input_acceleration
   - 0.199089527130127 x2_input_acceleration
   + 0.4983010292053223 x3_input_acceleration >= -0.0468205493318635
 CBF_Constraint_sphere_right: 0.2477891445159912 x1_input_acceleration
   - 0.6490890979766846 x2_input_acceleration
   + 0.2983012199401855 x3_input_acceleration >= -0.1122999337674889
 Relaxed_CLF_constraint: 0.00989490747451782 x1_input_acceleration
   - 0.0355443954467773 x2_input_acceleration
   + 0.0101504921913147 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2472866177558899 x1_input_acceleration
   + 0.2027361392974854 x2_input_acceleration
   + 0.2977864742279053 x3_input_acceleration >= -0.017107349432871
 CBF_Constraint_sphere_low: 0.1472868323326111 x1_input_acceleration
   - 0.1972634792327881 x2_input_acceleration
   + 0.497786283493042 x3_input_acceleration >= -0.0464743688128536
 CBF_Constraint_sphere_right: 0.2472867965698242 x1_input_acceleration
   - 0.6472630500793457 x2_input_acceleration
   + 0.2977864742279053 x3_input_acceleration >= -0.1115692499972175
 Relaxed_CLF_constraint: 0.00964373350143433 x1_input_acceleration
   - 0.0346313714981079 x2_input_acceleration
   + 0.00989311933517456 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2467842698097229 x1_input_acceleration
   + 0.2045621871948242 x2_input_acceleration
   + 0.297271728515625 x3_input_acceleration >= -0.017154661207529
 CBF_Constraint_sphere_low: 0.1467844843864441 x1_input_acceleration
   - 0.1954374313354492 x2_input_acceleration
   + 0.4972715377807617 x3_input_acceleration >= -0.0461301141776091
 CBF_Constraint_sphere_right: 0.2467844486236572 x1_input_acceleration
   - 0.6454370021820068 x2_input_acceleration
   + 0.297271728515625 x3_input_acceleration >= -0.1108404921107115
 Relaxed_CLF_constraint: 0.00939255952835083 x1_input_acceleration
   - 0.0337183475494385 x2_input_acceleration
   + 0.00963574647903442 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2462819218635559 x1_input_acceleration
   + 0.2063882350921631 x2_input_acceleration
   + 0.2967569828033447 x3_input_acceleration >= -0.0172038988659523
 CBF_Constraint_sphere_low: 0.1462821364402771 x1_input_acceleration
   - 0.1936113834381104 x2_input_acceleration
   + 0.4967567920684814 x3_input_acceleration >= -0.0457877854261298
 CBF_Constraint_sphere_right: 0.2462821006774902 x1_input_acceleration
   - 0.643610954284668 x2_input_acceleration
   + 0.2967569828033447 x3_input_acceleration >= -0.1101136601079708
 Relaxed_CLF_constraint: 0.00914138555526733 x1_input_acceleration
   - 0.032805323600769 x2_input_acceleration
   + 0.00937837362289429 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2457795739173889 x1_input_acceleration
   + 0.208214282989502 x2_input_acceleration
   + 0.2962422370910645 x3_input_acceleration >= -0.017255062408141
 CBF_Constraint_sphere_low: 0.1457797884941101 x1_input_acceleration
   - 0.1917853355407715 x2_input_acceleration
   + 0.4962420463562012 x3_input_acceleration >= -0.045447382558416
 CBF_Constraint_sphere_right: 0.2457797527313232 x1_input_acceleration
   - 0.6417849063873291 x2_input_acceleration
   + 0.2962422370910645 x3_input_acceleration >= -0.1093887539889954
 Relaxed_CLF_constraint: 0.00889021158218384 x1_input_acceleration
   - 0.0318922996520996 x2_input_acceleration
   + 0.00912100076675415 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2452772259712219 x1_input_acceleration
   + 0.2100403308868408 x2_input_acceleration
   + 0.2957276105880737 x3_input_acceleration >= -0.0173081694608307
 CBF_Constraint_sphere_low: 0.1452774405479431 x1_input_acceleration
   - 0.1899592876434326 x2_input_acceleration
   + 0.4957274198532104 x3_input_acceleration >= -0.0451089351221207
 CBF_Constraint_sphere_right: 0.2452774047851562 x1_input_acceleration
   - 0.6399588584899902 x2_input_acceleration
   + 0.2957276105880737 x3_input_acceleration >= -0.108665791380521
 Relaxed_CLF_constraint: 0.00863903760910034 x1_input_acceleration
   - 0.0309792757034302 x2_input_acceleration
   + 0.00886368751525879 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2447748780250549 x1_input_acceleration
   + 0.2118666172027588 x2_input_acceleration
   + 0.2952131032943726 x3_input_acceleration >= -0.0173632451884537
 CBF_Constraint_sphere_low: 0.1447750926017761 x1_input_acceleration
   - 0.1881330013275146 x2_input_acceleration
   + 0.4952129125595093 x3_input_acceleration >= -0.0447724205980059
 CBF_Constraint_sphere_right: 0.2447750568389893 x1_input_acceleration
   - 0.6381325721740723 x2_input_acceleration
   + 0.2952131032943726 x3_input_acceleration >= -0.1079446961191805
 Relaxed_CLF_constraint: 0.00838786363601685 x1_input_acceleration
   - 0.0300661325454712 x2_input_acceleration
   + 0.0086064338684082 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2442725300788879 x1_input_acceleration
   + 0.2136926651000977 x2_input_acceleration
   + 0.2946983575820923 x3_input_acceleration >= -0.0174201865075766
 CBF_Constraint_sphere_low: 0.1442727446556091 x1_input_acceleration
   - 0.1863069534301758 x2_input_acceleration
   + 0.494698166847229 x3_input_acceleration >= -0.0444377955072263
 CBF_Constraint_sphere_right: 0.2442727088928223 x1_input_acceleration
   - 0.6363065242767334 x2_input_acceleration
   + 0.2946983575820923 x3_input_acceleration >= -0.1072255677771393
 Relaxed_CLF_constraint: 0.00813668966293335 x1_input_acceleration
   - 0.0291531085968018 x2_input_acceleration
   + 0.00834906101226807 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2437700629234314 x1_input_acceleration
   + 0.2155187129974365 x2_input_acceleration
   + 0.2941837310791016 x3_input_acceleration >= -0.0174790567153466
 CBF_Constraint_sphere_low: 0.1437702775001526 x1_input_acceleration
   - 0.1844809055328369 x2_input_acceleration
   + 0.4941835403442383 x3_input_acceleration >= -0.0441051171864629
 CBF_Constraint_sphere_right: 0.2437702417373657 x1_input_acceleration
   - 0.6344804763793945 x2_input_acceleration
   + 0.2941837310791016 x3_input_acceleration >= -0.1065083683237346
 Relaxed_CLF_constraint: 0.00788545608520508 x1_input_acceleration
   - 0.0282400846481323 x2_input_acceleration
   + 0.00809174776077271 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.243267834186554 x1_input_acceleration
   + 0.2173447608947754 x2_input_acceleration
   + 0.2936688661575317 x3_input_acceleration >= -0.0175398467971185
 CBF_Constraint_sphere_low: 0.1432680487632751 x1_input_acceleration
   - 0.182654857635498 x2_input_acceleration
   + 0.4936686754226685 x3_input_acceleration >= -0.0437743229769631
 CBF_Constraint_sphere_right: 0.2432680130004883 x1_input_acceleration
   - 0.6326544284820557 x2_input_acceleration
   + 0.2936688661575317 x3_input_acceleration >= -0.1057930887443531
 Relaxed_CLF_constraint: 0.00763434171676636 x1_input_acceleration
   - 0.0273270606994629 x2_input_acceleration
   + 0.00783431529998779 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2427656054496765 x1_input_acceleration
   + 0.2191708087921143 x2_input_acceleration
   + 0.2931541204452515 x3_input_acceleration >= -0.0176025802374916
 CBF_Constraint_sphere_low: 0.1427658200263977 x1_input_acceleration
   - 0.1808288097381592 x2_input_acceleration
   + 0.4931539297103882 x3_input_acceleration >= -0.0434454840469819
 CBF_Constraint_sphere_right: 0.2427657842636108 x1_input_acceleration
   - 0.6308283805847168 x2_input_acceleration
   + 0.2931541204452515 x3_input_acceleration >= -0.1050797525235728
 Relaxed_CLF_constraint: 0.00738322734832764 x1_input_acceleration
   - 0.0264140367507935 x2_input_acceleration
   + 0.00757694244384766 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2422632575035095 x1_input_acceleration
   + 0.2209968566894531 x2_input_acceleration
   + 0.2926393747329712 x3_input_acceleration >= -0.0176672250617337
 CBF_Constraint_sphere_low: 0.1422634720802307 x1_input_acceleration
   - 0.1790027618408203 x2_input_acceleration
   + 0.4926391839981079 x3_input_acceleration >= -0.0431185624613214
 CBF_Constraint_sphere_right: 0.2422634363174438 x1_input_acceleration
   - 0.6290023326873779 x2_input_acceleration
   + 0.2926393747329712 x3_input_acceleration >= -0.1043683276866508
 Relaxed_CLF_constraint: 0.00713205337524414 x1_input_acceleration
   - 0.025501012802124 x2_input_acceleration
   + 0.00731956958770752 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2417609095573425 x1_input_acceleration
   + 0.222822904586792 x2_input_acceleration
   + 0.2921246290206909 x3_input_acceleration >= -0.0177337957697411
 CBF_Constraint_sphere_low: 0.1417611241340637 x1_input_acceleration
   - 0.1771767139434814 x2_input_acceleration
   + 0.4921244382858276 x3_input_acceleration >= -0.0427935667594263
 CBF_Constraint_sphere_right: 0.2417610883712769 x1_input_acceleration
   - 0.6271762847900391 x2_input_acceleration
   + 0.2921246290206909 x3_input_acceleration >= -0.1036588287334942
 Relaxed_CLF_constraint: 0.00688087940216064 x1_input_acceleration
   - 0.0245879888534546 x2_input_acceleration
   + 0.00706219673156738 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint_sphere_left: 0.2412585616111755 x1_input_acceleration
   + 0.2246489524841309 x2_input_acceleration
   + 0.2916098833084106 x3_input_acceleration >= -0.0178022923615139
 CBF_Constraint_sphere_low: 0.1412587761878967 x1_input_acceleration
   - 0.1753506660461426 x2_input_acceleration
   + 0.4916096925735474 x3_input_acceleration >= -0.0424704969412965
 CBF_Constraint_sphere_right: 0.2412587404251099 x1_input_acceleration
   - 0.6253502368927002 x2_input_acceleration
   + 0.2916098833084106 x3_input_acceleration >= -0.1029512556641029
 Relaxed_CLF_constraint: 0.00662970542907715 x1_input_acceleration
   - 0.0236749649047852 x2_input_acceleration
   + 0.00680482387542725 x3_input_acceleration + relaxation_CLF <= -20.8
Bounds
 -0.1 <= x1_input_acceleration <= 0.1
 -0.1 <= x2_input_acceleration <= 0.1
 -0.1 <= x3_input_acceleration <= 0.1
 -30 <= relaxation_CLF <= 30
End
