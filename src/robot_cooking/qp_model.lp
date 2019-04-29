\ Model CBF_CLF_QP
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
 [ 2 x1_input_acceleration ^2 + 2 x2_input_acceleration ^2
   + 2 x3_input_acceleration ^2 + 2 relaxation_CLF ^2 ] / 2 
Subject To
 CBF_Constraint_h_left: 0.010667338883467 x1_input_acceleration
   + 0.4541377671176097 x2_input_acceleration
   - 0.1156306634125268 x3_input_acceleration >= -0.044931338490661
 CBF_Constraint_h_left: 0.3906673388834671 x1_input_acceleration
   + 0.4541377671176097 x2_input_acceleration
   - 0.1156306634125268 x3_input_acceleration >= -0.0830581328785198
 CBF_Constraint_h_left: - 0.309332661116533 x1_input_acceleration
   + 0.4541377671176097 x2_input_acceleration
   - 0.1156306634125268 x3_input_acceleration >= -0.0688245642693063
 Relaxed_CLF_constraint: 0.0752366330037452 x1_input_acceleration
   + 0.3836571294274876 x2_input_acceleration
   - 0.0519329321472424 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: 0.00766733891126514 x1_input_acceleration
   + 0.451137767123957 x2_input_acceleration
   - 0.1160325910791847 x3_input_acceleration >= -0.0442619088010298
 CBF_Constraint_h_left: 0.3876673389112651 x1_input_acceleration
   + 0.451137767123957 x2_input_acceleration
   - 0.1160325910791847 x3_input_acceleration >= -0.0818187031941701
 CBF_Constraint_h_left: - 0.3123326610887349 x1_input_acceleration
   + 0.451137767123957 x2_input_acceleration
   - 0.1160325910791847 x3_input_acceleration >= -0.0686351345752274
 Relaxed_CLF_constraint: 0.0737366330176442 x1_input_acceleration
   + 0.3821571294306613 x2_input_acceleration
   - 0.0521338959805713 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: 0.004702328828324 x1_input_acceleration
   + 0.4481377671241643 x2_input_acceleration
   - 0.116448765311483 x3_input_acceleration >= -0.0436024712905026
 CBF_Constraint_h_left: 0.384702328828324 x1_input_acceleration
   + 0.4481377671241643 x2_input_acceleration
   - 0.116448765311483 x3_input_acceleration >= -0.0805959137678841
 CBF_Constraint_h_left: - 0.315297671171676 x1_input_acceleration
   + 0.4481377671241643 x2_input_acceleration
   - 0.116448765311483 x3_input_acceleration >= -0.0684500986779707
 Relaxed_CLF_constraint: 0.0722541279761736 x1_input_acceleration
   + 0.380657129430765 x2_input_acceleration
   - 0.0523419830967205 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: 0.00178179389651534 x1_input_acceleration
   + 0.4451377671242541 x2_input_acceleration
   - 0.1168787610963327 x3_input_acceleration >= -0.0429528628263175
 CBF_Constraint_h_left: 0.3817817938965153 x1_input_acceleration
   + 0.4451377671242541 x2_input_acceleration
   - 0.1168787610963327 x3_input_acceleration >= -0.0793914036666554
 CBF_Constraint_h_left: - 0.3182182061034847 x1_input_acceleration
   + 0.4451377671242541 x2_input_acceleration
   - 0.1168787610963327 x3_input_acceleration >= -0.068267775802875
 Relaxed_CLF_constraint: 0.0707938605102693 x1_input_acceleration
   + 0.3791571294308098 x2_input_acceleration
   - 0.0525569809891453 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.00109493301209257 x1_input_acceleration
   + 0.4421377671243166 x2_input_acceleration
   - 0.1173232657320532 x3_input_acceleration >= -0.0423129381695028
 CBF_Constraint_h_left: 0.3789050669879074 x1_input_acceleration
   + 0.4421377671243166 x2_input_acceleration
   - 0.1173232657320532 x3_input_acceleration >= -0.0782049008972052
 CBF_Constraint_h_left: - 0.3210949330120926 x1_input_acceleration
   + 0.4421377671243166 x2_input_acceleration
   - 0.1173232657320532 x3_input_acceleration >= -0.0680881274514377
 Relaxed_CLF_constraint: 0.0693554970559654 x1_input_acceleration
   + 0.3776571294308411 x2_input_acceleration
   - 0.0527792333070056 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.00392850901403552 x1_input_acceleration
   + 0.4391377671334012 x2_input_acceleration
   - 0.1177828726563203 x3_input_acceleration >= -0.0416825541992894
 CBF_Constraint_h_left: 0.3760714909859645 x1_input_acceleration
   + 0.4391377671334012 x2_input_acceleration
   - 0.1177828726563203 x3_input_acceleration >= -0.0770361374866227
 CBF_Constraint_h_left: - 0.3239285090140355 x1_input_acceleration
   + 0.4391377671334012 x2_input_acceleration
   - 0.1177828726563203 x3_input_acceleration >= -0.0679111156415351
 Relaxed_CLF_constraint: 0.0679387090549939 x1_input_acceleration
   + 0.3761571294353834 x2_input_acceleration
   - 0.0530090367691392 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.00671958136617046 x1_input_acceleration
   + 0.4361377671360258 x2_input_acceleration
   - 0.118258085400723 x3_input_acceleration >= -0.0410615698646949
 CBF_Constraint_h_left: 0.3732804186338295 x1_input_acceleration
   + 0.4361377671360258 x2_input_acceleration
   - 0.118258085400723 x3_input_acceleration >= -0.0758848494051225
 CBF_Constraint_h_left: - 0.3267195813661705 x1_input_acceleration
   + 0.4361377671360258 x2_input_acceleration
   - 0.118258085400723 x3_input_acceleration >= -0.0677367028832822
 Relaxed_CLF_constraint: 0.0665431728789264 x1_input_acceleration
   + 0.3746571294366957 x2_input_acceleration
   - 0.0532466431413405 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.00946878764418208 x1_input_acceleration
   + 0.4331377671368799 x2_input_acceleration
   - 0.1187493219171706 x3_input_acceleration >= -0.0404498461788901
 CBF_Constraint_h_left: 0.3705312123558179 x1_input_acceleration
   + 0.4331377671368799 x2_input_acceleration
   - 0.1187493219171706 x3_input_acceleration >= -0.0747507765264955
 CBF_Constraint_h_left: - 0.3294687876441821 x1_input_acceleration
   + 0.4331377671368799 x2_input_acceleration
   - 0.1187493219171706 x3_input_acceleration >= -0.0675648522019592
 Relaxed_CLF_constraint: 0.0651685697399206 x1_input_acceleration
   + 0.3731571294371228 x2_input_acceleration
   - 0.0534922613995643 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0121767558312516 x1_input_acceleration
   + 0.4301377671370889 x2_input_acceleration
   - 0.1192569184248171 x3_input_acceleration >= -0.0398472461731095
 CBF_Constraint_h_left: 0.3678232441687485 x1_input_acceleration
   + 0.4301377671370889 x2_input_acceleration
   - 0.1192569184248171 x3_input_acceleration >= -0.0736336625651717
 CBF_Constraint_h_left: - 0.3321767558312516 x1_input_acceleration
   + 0.4301377671370889 x2_input_acceleration
   - 0.1192569184248171 x3_input_acceleration >= -0.0673955271061097
 Relaxed_CLF_constraint: 0.0638145856463859 x1_input_acceleration
   + 0.3716571294372273 x2_input_acceleration
   - 0.0537460596533875 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0145613400590172 x1_input_acceleration
   + 0.4278612483154448 x2_input_acceleration
   - 0.1185718743701526 x3_input_acceleration >= -0.0393341424565041
 CBF_Constraint_h_left: 0.3651558955061684 x1_input_acceleration
   + 0.4271377671371611 x2_input_acceleration
   - 0.1197811334033557 x3_input_acceleration >= -0.072533255014306
 CBF_Constraint_h_left: - 0.3348441044938316 x1_input_acceleration
   + 0.4271377671371611 x2_input_acceleration
   - 0.1197811334033557 x3_input_acceleration >= -0.0672286915871471
 Relaxed_CLF_constraint: 0.0624809113150958 x1_input_acceleration
   + 0.3701571294372634 x2_input_acceleration
   - 0.0540081671426568 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0171917430422618 x1_input_acceleration
   + 0.4248612483174034 x2_input_acceleration
   - 0.1191222348956373 x3_input_acceleration >= -0.0387481857992962
 CBF_Constraint_h_left: 0.3625254925229237 x1_input_acceleration
   + 0.4241377671391197 x2_input_acceleration
   - 0.1203314939288404 x3_input_acceleration >= -0.0714493116684733
 CBF_Constraint_h_left: - 0.3374745074770763 x1_input_acceleration
   + 0.4241377671391197 x2_input_acceleration
   - 0.1203314939288404 x3_input_acceleration >= -0.06706538928545
 Relaxed_CLF_constraint: 0.0611657098234735 x1_input_acceleration
   + 0.3686571294382427 x2_input_acceleration
   - 0.0542833474053992 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0197828563388921 x1_input_acceleration
   + 0.4218612483176931 x2_input_acceleration
   - 0.1196896555552207 x3_input_acceleration >= -0.0381709719710037
 CBF_Constraint_h_left: 0.3602171436611079 x1_input_acceleration
   + 0.4218612483176931 x2_input_acceleration
   - 0.1196896555552207 x3_input_acceleration >= -0.0705122292666142
 CBF_Constraint_h_left: - 0.3397828563388921 x1_input_acceleration
   + 0.4218612483176931 x2_input_acceleration
   - 0.1196896555552207 x3_input_acceleration >= -0.0669362289852265
 Relaxed_CLF_constraint: 0.0598701531751584 x1_input_acceleration
   + 0.3671571294383875 x2_input_acceleration
   - 0.0545670577351909 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0223361134939086 x1_input_acceleration
   + 0.4188612483177445 x2_input_acceleration
   - 0.1202740324262034 x3_input_acceleration >= -0.0376023725460928
 CBF_Constraint_h_left: 0.3576638865060914 x1_input_acceleration
   + 0.4188612483177445 x2_input_acceleration
   - 0.1202740324262034 x3_input_acceleration >= -0.0694585109822502
 CBF_Constraint_h_left: - 0.3423361134939086 x1_input_acceleration
   + 0.4188612483177445 x2_input_acceleration
   - 0.1202740324262034 x3_input_acceleration >= -0.0667761507051182
 Relaxed_CLF_constraint: 0.0585935245976501 x1_input_acceleration
   + 0.3656571294384132 x2_input_acceleration
   - 0.0548592461706822 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0248510717917064 x1_input_acceleration
   + 0.4158612483178394 x2_input_acceleration
   - 0.1208754770880836 x3_input_acceleration >= -0.037042258645735
 CBF_Constraint_h_left: 0.3551489282082936 x1_input_acceleration
   + 0.4158612483178394 x2_input_acceleration
   - 0.1208754770880836 x3_input_acceleration >= -0.0684205550053108
 CBF_Constraint_h_left: - 0.3448510717917064 x1_input_acceleration
   + 0.4158612483178394 x2_input_acceleration
   - 0.1208754770880836 x3_input_acceleration >= -0.066618430132408
 Relaxed_CLF_constraint: 0.0573360454487512 x1_input_acceleration
   + 0.3641571294384607 x2_input_acceleration
   - 0.0551599685016223 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0273283057353306 x1_input_acceleration
   + 0.4128612483277141 x2_input_acceleration
   - 0.1214939158394806 x3_input_acceleration >= -0.0364905045627732
 CBF_Constraint_h_left: 0.3526716942646694 x1_input_acceleration
   + 0.4128612483277141 x2_input_acceleration
   - 0.1214939158394806 x3_input_acceleration >= -0.0673981264730604
 CBF_Constraint_h_left: - 0.3473283057353306 x1_input_acceleration
   + 0.4128612483277141 x2_input_acceleration
   - 0.1214939158394806 x3_input_acceleration >= -0.0664630334804261
 Relaxed_CLF_constraint: 0.0560974284769391 x1_input_acceleration
   + 0.362657129443398 x2_input_acceleration
   - 0.0554691878773208 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0297683811510633 x1_input_acceleration
   + 0.4098612483278842 x2_input_acceleration
   - 0.1221292191093438 x3_input_acceleration >= -0.0359469863893761
 CBF_Constraint_h_left: 0.3502316188489367 x1_input_acceleration
   + 0.4098612483278842 x2_input_acceleration
   - 0.1221292191093438 x3_input_acceleration >= -0.0663909939706741
 CBF_Constraint_h_left: - 0.3497683811510633 x1_input_acceleration
   + 0.4098612483278842 x2_input_acceleration
   - 0.1221292191093438 x3_input_acceleration >= -0.0663099273735463
 Relaxed_CLF_constraint: 0.0548773907690728 x1_input_acceleration
   + 0.361157129443483 x2_input_acceleration
   - 0.0557868395122524 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0321718554338641 x1_input_acceleration
   + 0.4068612483279381 x2_input_acceleration
   - 0.1227812057134932 x3_input_acceleration >= -0.0354115820373712
 CBF_Constraint_h_left: 0.3478281445661359 x1_input_acceleration
   + 0.4068612483279381 x2_input_acceleration
   - 0.1227812057134932 x3_input_acceleration >= -0.065398929504937
 CBF_Constraint_h_left: - 0.3521718554338641 x1_input_acceleration
   + 0.4068612483279381 x2_input_acceleration
   - 0.1227812057134932 x3_input_acceleration >= -0.0661590789067894
 Relaxed_CLF_constraint: 0.0536756536276723 x1_input_acceleration
   + 0.35965712944351 x2_input_acceleration
   - 0.0561128328143271 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0345435691660628 x1_input_acceleration
   + 0.4038667319746084 x2_input_acceleration
   - 0.1234556068122541 x3_input_acceleration >= -0.0348857205549907
 CBF_Constraint_h_left: 0.345460722368361 x1_input_acceleration
   + 0.403861248347289 x2_input_acceleration
   - 0.1234496463477766 x3_input_acceleration >= -0.0644217084498229
 CBF_Constraint_h_left: - 0.354539277631639 x1_input_acceleration
   + 0.403861248347289 x2_input_acceleration
   - 0.1234496463477766 x3_input_acceleration >= -0.0660104556208966
 Relaxed_CLF_constraint: 0.0524919425287849 x1_input_acceleration
   + 0.3581571294531855 x2_input_acceleration
   - 0.0564470531314688 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0368753639085302 x1_input_acceleration
   + 0.400866731974908 x2_input_acceleration
   - 0.1241399019092276 x3_input_acceleration >= -0.0343661611284155
 CBF_Constraint_h_left: 0.3431246360914698 x1_input_acceleration
   + 0.400866731974908 x2_input_acceleration
   - 0.1241399019092276 x3_input_acceleration >= -0.0634598419857947
 CBF_Constraint_h_left: - 0.3568753639085303 x1_input_acceleration
   + 0.400866731974908 x2_input_acceleration
   - 0.1241399019092276 x3_input_acceleration >= -0.0658662193537803
 Relaxed_CLF_constraint: 0.0513260451575512 x1_input_acceleration
   + 0.3566571294533353 x2_input_acceleration
   - 0.0567892006799556 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0391722334746699 x1_input_acceleration
   + 0.3978667319897152 x2_input_acceleration
   - 0.1248400593910696 x3_input_acceleration >= -0.0338543601820839
 CBF_Constraint_h_left: 0.3408277665253301 x1_input_acceleration
   + 0.3978667319897152 x2_input_acceleration
   - 0.1248400593910696 x3_input_acceleration >= -0.0625116358218966
 CBF_Constraint_h_left: - 0.3591722334746699 x1_input_acceleration
   + 0.3978667319897152 x2_input_acceleration
   - 0.1248400593910696 x3_input_acceleration >= -0.0657219175380311
 Relaxed_CLF_constraint: 0.0501776103744814 x1_input_acceleration
   + 0.3551571294607389 x2_input_acceleration
   - 0.0571392794208766 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0414255900640211 x1_input_acceleration
   + 0.3948705466907259 x2_input_acceleration
   - 0.1255338044802909 x3_input_acceleration >= -0.0333493910558202
 CBF_Constraint_h_left: 0.3385653500299731 x1_input_acceleration
   + 0.3948667319934602 x2_input_acceleration
   - 0.1255557389895683 x3_input_acceleration >= -0.0615776189673325
 CBF_Constraint_h_left: - 0.3614346499700269 x1_input_acceleration
   + 0.3948667319934602 x2_input_acceleration
   - 0.1255557389895683 x3_input_acceleration >= -0.0655797464568419
 Relaxed_CLF_constraint: 0.0490464021268029 x1_input_acceleration
   + 0.3536571294626114 x2_input_acceleration
   - 0.0574971192201259 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0436542598404365 x1_input_acceleration
   + 0.391870546693281 x2_input_acceleration
   - 0.1262651290080293 x3_input_acceleration >= -0.0328527756428304
 CBF_Constraint_h_left: 0.3363457401595635 x1_input_acceleration
   + 0.391870546693281 x2_input_acceleration
   - 0.1262651290080293 x3_input_acceleration >= -0.0606584662731474
 CBF_Constraint_h_left: - 0.3636542598404365 x1_input_acceleration
   + 0.391870546693281 x2_input_acceleration
   - 0.1262651290080293 x3_input_acceleration >= -0.0654374572173002
 Relaxed_CLF_constraint: 0.0479320672385951 x1_input_acceleration
   + 0.352157129463889 x2_input_acceleration
   - 0.0578627814839951 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0458494459458686 x1_input_acceleration
   + 0.3888705466937305 x2_input_acceleration
   - 0.1270111714734062 x3_input_acceleration >= -0.0323635778646177
 CBF_Constraint_h_left: 0.3341505540541314 x1_input_acceleration
   + 0.3888705466937305 x2_input_acceleration
   - 0.1270111714734062 x3_input_acceleration >= -0.0597521831349027
 CBF_Constraint_h_left: - 0.3658494459458686 x1_input_acceleration
   + 0.3888705466937305 x2_input_acceleration
   - 0.1270111714734062 x3_input_acceleration >= -0.0652994892159567
 Relaxed_CLF_constraint: 0.0468344741858791 x1_input_acceleration
   + 0.3506571294641138 x2_input_acceleration
   - 0.0582358027166835 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0480117042685111 x1_input_acceleration
   + 0.3858705467008098 x2_input_acceleration
   - 0.1277714665381375 x3_input_acceleration >= -0.0318816875548138
 CBF_Constraint_h_left: 0.3319882957314889 x1_input_acceleration
   + 0.3858705467008098 x2_input_acceleration
   - 0.1277714665381375 x3_input_acceleration >= -0.0588594637437967
 CBF_Constraint_h_left: - 0.3680117042685111 x1_input_acceleration
   + 0.3858705467008098 x2_input_acceleration
   - 0.1277714665381375 x3_input_acceleration >= -0.0651635602377756
 Relaxed_CLF_constraint: 0.0457533450245579 x1_input_acceleration
   + 0.3491571294676534 x2_input_acceleration
   - 0.0586159502490492 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0501415287046492 x1_input_acceleration
   + 0.3828705467009096 x2_input_acceleration
   - 0.1285455400874644 x3_input_acceleration >= -0.0314069960770676
 CBF_Constraint_h_left: 0.3298584712953508 x1_input_acceleration
   + 0.3828705467009096 x2_input_acceleration
   - 0.1285455400874644 x3_input_acceleration >= -0.0579801056231843
 CBF_Constraint_h_left: - 0.3701415287046492 x1_input_acceleration
   + 0.3828705467009096 x2_input_acceleration
   - 0.1285455400874644 x3_input_acceleration >= -0.0650296406698115
 Relaxed_CLF_constraint: 0.0446884328064888 x1_input_acceleration
   + 0.3476571294677033 x2_input_acceleration
   - 0.0590029870237126 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.052239405775122 x1_input_acceleration
   + 0.3798705467009321 x2_input_acceleration
   - 0.1293328952518908 x3_input_acceleration >= -0.0309393963902099
 CBF_Constraint_h_left: 0.327760594224878 x1_input_acceleration
   + 0.3798705467009321 x2_input_acceleration
   - 0.1293328952518908 x3_input_acceleration >= -0.0571139092929367
 CBF_Constraint_h_left: - 0.372239405775122 x1_input_acceleration
   + 0.3798705467009321 x2_input_acceleration
   - 0.1293328952518908 x3_input_acceleration >= -0.0648977013142294
 Relaxed_CLF_constraint: 0.0436394942712524 x1_input_acceleration
   + 0.3461571294677145 x2_input_acceleration
   - 0.0593966646059258 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0543110598979537 x1_input_acceleration
   + 0.3768625596787571 x2_input_acceleration
   - 0.1301452933362471 x3_input_acceleration >= -0.0304782193731104
 CBF_Constraint_h_left: 0.3256941853107865 x1_input_acceleration
   + 0.376870546701157 x2_input_acceleration
   - 0.1301330147794233 x3_input_acceleration >= -0.0562606782129169
 CBF_Constraint_h_left: - 0.3743058146892135 x1_input_acceleration
   + 0.376870546701157 x2_input_acceleration
   - 0.1301330147794233 x3_input_acceleration >= -0.0647677133541416
 Relaxed_CLF_constraint: 0.0426062898142067 x1_input_acceleration
   + 0.344657129467827 x2_input_acceleration
   - 0.0597967243696921 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.056346572616823 x1_input_acceleration
   + 0.373862559678858 x2_input_acceleration
   - 0.1309578922805995 x3_input_acceleration >= -0.0300245298314669
 CBF_Constraint_h_left: 0.323653427383177 x1_input_acceleration
   + 0.373862559678858 x2_input_acceleration
   - 0.1309578922805995 x3_input_acceleration >= -0.0554186810342705
 CBF_Constraint_h_left: - 0.376346572616823 x1_input_acceleration
   + 0.373862559678858 x2_input_acceleration
   - 0.1309578922805995 x3_input_acceleration >= -0.0646399814501586
 Relaxed_CLF_constraint: 0.041588533454772 x1_input_acceleration
   + 0.3431571294678775 x2_input_acceleration
   - 0.0602030238418683 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0583517794097547 x1_input_acceleration
   + 0.3708625596813199 x2_input_acceleration
   - 0.1317827322246976 x3_input_acceleration >= -0.0295776642115679
 CBF_Constraint_h_left: 0.3216482205902453 x1_input_acceleration
   + 0.3708625596813199 x2_input_acceleration
   - 0.1317827322246976 x3_input_acceleration >= -0.0545908261237145
 CBF_Constraint_h_left: - 0.3783517794097547 x1_input_acceleration
   + 0.3708625596813199 x2_input_acceleration
   - 0.1317827322246976 x3_input_acceleration >= -0.0645139489171286
 Relaxed_CLF_constraint: 0.0405859300583062 x1_input_acceleration
   + 0.3416571294691084 x2_input_acceleration
   - 0.0606154438139174 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0603273894379603 x1_input_acceleration
   + 0.3678625596896158 x2_input_acceleration
   - 0.1326198248650103 x3_input_acceleration >= -0.0291375686712554
 CBF_Constraint_h_left: 0.3196726105620397 x1_input_acceleration
   + 0.3678625596896158 x2_input_acceleration
   - 0.1326198248650103 x3_input_acceleration >= -0.0537753646780429
 CBF_Constraint_h_left: - 0.3803273894379603 x1_input_acceleration
   + 0.3678625596896158 x2_input_acceleration
   - 0.1326198248650103 x3_input_acceleration >= -0.064389950981329
 Relaxed_CLF_constraint: 0.0395981250442034 x1_input_acceleration
   + 0.3401571294732564 x2_input_acceleration
   - 0.0610339901340737 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0622740404081133 x1_input_acceleration
   + 0.3648625596904784 x2_input_acceleration
   - 0.1334690021027552 x3_input_acceleration >= -0.0287041795237361
 CBF_Constraint_h_left: 0.3177259595918867 x1_input_acceleration
   + 0.3648625596904784 x2_input_acceleration
   - 0.1334690021027552 x3_input_acceleration >= -0.0529721118461946
 CBF_Constraint_h_left: - 0.3822740404081133 x1_input_acceleration
   + 0.3648625596904784 x2_input_acceleration
   - 0.1334690021027552 x3_input_acceleration >= -0.0642680259890342
 Relaxed_CLF_constraint: 0.0386247995591269 x1_input_acceleration
   + 0.3386571294736876 x2_input_acceleration
   - 0.0614585787529462 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0641923572794717 x1_input_acceleration
   + 0.3618625596920123 x2_input_acceleration
   - 0.1343300514210049 x3_input_acceleration >= -0.0282774333886801
 CBF_Constraint_h_left: 0.3158076427205283 x1_input_acceleration
   + 0.3618625596920123 x2_input_acceleration
   - 0.1343300514210049 x3_input_acceleration >= -0.0521808855055805
 CBF_Constraint_h_left: - 0.3841923572794717 x1_input_acceleration
   + 0.3618625596920123 x2_input_acceleration
   - 0.1343300514210049 x3_input_acceleration >= -0.0641482105533956
 Relaxed_CLF_constraint: 0.0376656411234477 x1_input_acceleration
   + 0.3371571294744546 x2_input_acceleration
   - 0.061889103412071 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0660878319786961 x1_input_acceleration
   + 0.3588595794616266 x2_input_acceleration
   - 0.1352047687547091 x3_input_acceleration >= -0.0278570322002585
 CBF_Constraint_h_left: 0.3139121680213039 x1_input_acceleration
   + 0.3588595794616266 x2_input_acceleration
   - 0.1352047687547091 x3_input_acceleration >= -0.0514003441243063
 CBF_Constraint_h_left: - 0.3860830636071141 x1_input_acceleration
   + 0.3588625596938653 x2_input_acceleration
   - 0.1352029806153658 x3_input_acceleration >= -0.0640305786803917
 Relaxed_CLF_constraint: 0.0367202879596265 x1_input_acceleration
   + 0.3356571294753811 x2_input_acceleration
   - 0.0623255680092515 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0679515016761497 x1_input_acceleration
   + 0.3558595794617666 x2_input_acceleration
   - 0.1360893308239043 x3_input_acceleration >= -0.0274434382097118
 CBF_Constraint_h_left: 0.3120484983238503 x1_input_acceleration
   + 0.3558595794617666 x2_input_acceleration
   - 0.1360893308239043 x3_input_acceleration >= -0.0506326528912434
 CBF_Constraint_h_left: - 0.3879515016761497 x1_input_acceleration
   + 0.3558595794617666 x2_input_acceleration
   - 0.1360893308239043 x3_input_acceleration >= -0.0639156784778957
 Relaxed_CLF_constraint: 0.0357884531108997 x1_input_acceleration
   + 0.3341571294754511 x2_input_acceleration
   - 0.0627678490438491 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0700682219512423 x1_input_acceleration
   + 0.3521194089830346 x2_input_acceleration
   - 0.1381991497516082 x3_input_acceleration >= -0.0269991597255094
 CBF_Constraint_h_left: 0.3099317780487577 x1_input_acceleration
   + 0.3521194089830346 x2_input_acceleration
   - 0.1381991497516082 x3_input_acceleration >= -0.0497861975547734
 CBF_Constraint_h_left: - 0.3900682219512423 x1_input_acceleration
   + 0.3521194089830346 x2_input_acceleration
   - 0.1381991497516082 x3_input_acceleration >= -0.0638100752377082
 Relaxed_CLF_constraint: 0.0348698062607069 x1_input_acceleration
   + 0.3326571294754955 x2_input_acceleration
   - 0.0632158044099532 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0718828497963378 x1_input_acceleration
   + 0.3491194089925249 x2_input_acceleration
   - 0.1391049667601993 x3_input_acceleration >= -0.0266004244018722
 CBF_Constraint_h_left: 0.3081171502036622 x1_input_acceleration
   + 0.3491194089925249 x2_input_acceleration
   - 0.1391049667601993 x3_input_acceleration >= -0.049042682940568
 CBF_Constraint_h_left: - 0.3918828497963378 x1_input_acceleration
   + 0.3491194089925249 x2_input_acceleration
   - 0.1391049667601993 x3_input_acceleration >= -0.0637016803692863
 Relaxed_CLF_constraint: 0.0339624923381592 x1_input_acceleration
   + 0.3311571294802407 x2_input_acceleration
   - 0.0636687129142488 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0736726146936583 x1_input_acceleration
   + 0.3461194089927422 x2_input_acceleration
   - 0.1400217923887019 x3_input_acceleration >= -0.0262081004452575
 CBF_Constraint_h_left: 0.3063273853063417 x1_input_acceleration
   + 0.3461194089927422 x2_input_acceleration
   - 0.1400217923887019 x3_input_acceleration >= -0.0483103036534625
 CBF_Constraint_h_left: - 0.3936726146936583 x1_input_acceleration
   + 0.3461194089927422 x2_input_acceleration
   - 0.1400217923887019 x3_input_acceleration >= -0.0635957187962428
 Relaxed_CLF_constraint: 0.0330676098894989 x1_input_acceleration
   + 0.3296571294803493 x2_input_acceleration
   - 0.0641271257285001 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0754381415483314 x1_input_acceleration
   + 0.343119408997592 x2_input_acceleration
   - 0.1409493504698814 x3_input_acceleration >= -0.0258221403572511
 CBF_Constraint_h_left: 0.3045618584516686 x1_input_acceleration
   + 0.343119408997592 x2_input_acceleration
   - 0.1409493504698814 x3_input_acceleration >= -0.0475888934630681
 CBF_Constraint_h_left: - 0.3954381415483314 x1_input_acceleration
   + 0.343119408997592 x2_input_acceleration
   - 0.1409493504698814 x3_input_acceleration >= -0.0634922430049841
 Relaxed_CLF_constraint: 0.0321848464621624 x1_input_acceleration
   + 0.3281571294827742 x2_input_acceleration
   - 0.0645909047690898 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0771705198801331 x1_input_acceleration
   + 0.3401377672283287 x2_input_acceleration
   - 0.141888793929777 x3_input_acceleration >= -0.0254453549191224
 CBF_Constraint_h_left: 0.3028294801198669 x1_input_acceleration
   + 0.3401377672283287 x2_input_acceleration
   - 0.141888793929777 x3_input_acceleration >= -0.0468829561418971
 CBF_Constraint_h_left: - 0.3971705198801331 x1_input_acceleration
   + 0.3401377672283287 x2_input_acceleration
   - 0.141888793929777 x3_input_acceleration >= -0.0633926380999437
 Relaxed_CLF_constraint: 0.0313138889246795 x1_input_acceleration
   + 0.3266571294828471 x2_input_acceleration
   - 0.0650599112433004 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0788896797465692 x1_input_acceleration
   + 0.3371377672332364 x2_input_acceleration
   - 0.1428369795255403 x3_input_acceleration >= -0.0250719645963769
 CBF_Constraint_h_left: 0.3011103202534308 x1_input_acceleration
   + 0.3371377672332364 x2_input_acceleration
   - 0.1428369795255403 x3_input_acceleration >= -0.0461829254445288
 CBF_Constraint_h_left: - 0.3988896797465692 x1_input_acceleration
   + 0.3371377672332364 x2_input_acceleration
   - 0.1428369795255403 x3_input_acceleration >= -0.063294313355828
 Relaxed_CLF_constraint: 0.0304543089914614 x1_input_acceleration
   + 0.325157129485301 x2_input_acceleration
   - 0.065534004041182 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0805865869727637 x1_input_acceleration
   + 0.3341377672332829 x2_input_acceleration
   - 0.1437952875012662 x3_input_acceleration >= -0.0247048325497835
 CBF_Constraint_h_left: 0.2994134130272363 x1_input_acceleration
   + 0.3341377672332829 x2_input_acceleration
   - 0.1437952875012662 x3_input_acceleration >= -0.0454933810249585
 CBF_Constraint_h_left: - 0.4005865869727637 x1_input_acceleration
   + 0.3341377672332829 x2_input_acceleration
   - 0.1437952875012662 x3_input_acceleration >= -0.0631986864654257
 Relaxed_CLF_constraint: 0.0296058553783642 x1_input_acceleration
   + 0.3236571294853243 x2_input_acceleration
   - 0.0660131580290449 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0822618715116363 x1_input_acceleration
   + 0.3311377672333798 x2_input_acceleration
   - 0.1447634314164383 x3_input_acceleration >= -0.0243439218670917
 CBF_Constraint_h_left: 0.2977381284883637 x1_input_acceleration
   + 0.3311377672333798 x2_input_acceleration
   - 0.1447634314164383 x3_input_acceleration >= -0.0448141662798808
 CBF_Constraint_h_left: - 0.4022618715116363 x1_input_acceleration
   + 0.3311377672333798 x2_input_acceleration
   - 0.1447634314164383 x3_input_acceleration >= -0.0631058213089535
 Relaxed_CLF_constraint: 0.0287682131089279 x1_input_acceleration
   + 0.3221571294853727 x2_input_acceleration
   - 0.066497229986631 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0836353067164823 x1_input_acceleration
   + 0.3288616060396562 x2_input_acceleration
   - 0.1445331767584095 x3_input_acceleration >= -0.0240086649101049
 CBF_Constraint_h_left: 0.2960838361973361 x1_input_acceleration
   + 0.3281377672335039 x2_input_acceleration
   - 0.1457411244894276 x3_input_acceleration >= -0.0441451269274407
 CBF_Constraint_h_left: - 0.4039161638026639 x1_input_acceleration
   + 0.3281377672335039 x2_input_acceleration
   - 0.1457411244894276 x3_input_acceleration >= -0.0630157842583731
 Relaxed_CLF_constraint: 0.0279410669634141 x1_input_acceleration
   + 0.3206571294854347 x2_input_acceleration
   - 0.0669860765231256 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0852692374717602 x1_input_acceleration
   + 0.3258616060429037 x2_input_acceleration
   - 0.1455201321034513 x3_input_acceleration >= -0.0236581844998205
 CBF_Constraint_h_left: 0.2944499054420582 x1_input_acceleration
   + 0.3251377672367514 x2_input_acceleration
   - 0.1467280798344693 x3_input_acceleration >= -0.0434861109776118
 CBF_Constraint_h_left: - 0.4055500945579418 x1_input_acceleration
   + 0.3251377672367514 x2_input_acceleration
   - 0.1467280798344693 x3_input_acceleration >= -0.0629286440728915
 Relaxed_CLF_constraint: 0.0271241015857751 x1_input_acceleration
   + 0.3191571294870585 x2_input_acceleration
   - 0.0674795541956465 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0868834788297136 x1_input_acceleration
   + 0.322861606045196 x2_input_acceleration
   - 0.1465161458753854 x3_input_acceleration >= -0.0233138341384535
 CBF_Constraint_h_left: 0.2931165211702864 x1_input_acceleration
   + 0.322861606045196 x2_input_acceleration
   - 0.1465161458753854 x3_input_acceleration >= -0.0429059731608079
 CBF_Constraint_h_left: - 0.4068834788297136 x1_input_acceleration
   + 0.322861606045196 x2_input_acceleration
   - 0.1465161458753854 x3_input_acceleration >= -0.0628151907512076
 Relaxed_CLF_constraint: 0.0263169809067984 x1_input_acceleration
   + 0.3176571294882046 x2_input_acceleration
   - 0.0679775610816136 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0884738339883655 x1_input_acceleration
   + 0.3198616060454635 x2_input_acceleration
   - 0.1475233044660364 x3_input_acceleration >= -0.0229755479207908
 CBF_Constraint_h_left: 0.2915261660116345 x1_input_acceleration
   + 0.3198616060454635 x2_input_acceleration
   - 0.1475233044660364 x3_input_acceleration >= -0.0422655194630013
 CBF_Constraint_h_left: - 0.4084738339883655 x1_input_acceleration
   + 0.3198616060454635 x2_input_acceleration
   - 0.1475233044660364 x3_input_acceleration >= -0.0627313613589293
 Relaxed_CLF_constraint: 0.0255218033274724 x1_input_acceleration
   + 0.3161571294883384 x2_input_acceleration
   - 0.0684811403769391 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.090045679727239 x1_input_acceleration
   + 0.3168616060455538 x2_input_acceleration
   - 0.1485390992019391 x3_input_acceleration >= -0.0226433414537579
 CBF_Constraint_h_left: 0.289954320272761 x1_input_acceleration
   + 0.3168616060455538 x2_input_acceleration
   - 0.1485390992019391 x3_input_acceleration >= -0.0416346623055825
 CBF_Constraint_h_left: - 0.410045679727239 x1_input_acceleration
   + 0.3168616060455538 x2_input_acceleration
   - 0.1485390992019391 x3_input_acceleration >= -0.0626506502101162
 Relaxed_CLF_constraint: 0.0247358804580357 x1_input_acceleration
   + 0.3146571294883835 x2_input_acceleration
   - 0.0689890377448904 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0915996466216829 x1_input_acceleration
   + 0.3138616060766147 x2_input_acceleration
   - 0.1495632362266823 x3_input_acceleration >= -0.0223171911652019
 CBF_Constraint_h_left: 0.2884003533783171 x1_input_acceleration
   + 0.3138616060766147 x2_input_acceleration
   - 0.1495632362266823 x3_input_acceleration >= -0.0410132583070822
 CBF_Constraint_h_left: - 0.4115996466216829 x1_input_acceleration
   + 0.3138616060766147 x2_input_acceleration
   - 0.1495632362266823 x3_input_acceleration >= -0.0625731346246712
 Relaxed_CLF_constraint: 0.0239588970108137 x1_input_acceleration
   + 0.313157129503914 x2_input_acceleration
   - 0.069501106257262 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0934138832903004 x1_input_acceleration
   + 0.3101233429534108 x2_input_acceleration
   - 0.1518166011675472 x3_input_acceleration >= -0.0219877304565097
 CBF_Constraint_h_left: 0.2865861167096996 x1_input_acceleration
   + 0.3101233429534108 x2_input_acceleration
   - 0.1518166011675472 x3_input_acceleration >= -0.0403390926313526
 CBF_Constraint_h_left: - 0.4134138832903004 x1_input_acceleration
   + 0.3101233429534108 x2_input_acceleration
   - 0.1518166011675472 x3_input_acceleration >= -0.0625339517829578
 Relaxed_CLF_constraint: 0.0231905382895421 x1_input_acceleration
   + 0.311657129507406 x2_input_acceleration
   - 0.0700171987466154 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0949592500376079 x1_input_acceleration
   + 0.3071320452611678 x2_input_acceleration
   - 0.1528252626655286 x3_input_acceleration >= -0.0216757283257002
 CBF_Constraint_h_left: 0.2850603002858785 x1_input_acceleration
   + 0.3071233429830306 x2_input_acceleration
   - 0.1528538728950208 x3_input_acceleration >= -0.0397371072657886
 CBF_Constraint_h_left: - 0.4149396997141215 x1_input_acceleration
   + 0.3071233429830306 x2_input_acceleration
   - 0.1528538728950208 x3_input_acceleration >= -0.0624660021657312
 Relaxed_CLF_constraint: 0.0224276300776315 x1_input_acceleration
   + 0.3101571295222159 x2_input_acceleration
   - 0.0705358346103522 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0964692169022203 x1_input_acceleration
   + 0.304132045268011 x2_input_acceleration
   - 0.1538699756292546 x3_input_acceleration >= -0.0213696450421946
 CBF_Constraint_h_left: 0.2835307830977797 x1_input_acceleration
   + 0.304132045268011 x2_input_acceleration
   - 0.1538699756292546 x3_input_acceleration >= -0.0391404938307728
 CBF_Constraint_h_left: - 0.4164692169022203 x1_input_acceleration
   + 0.304132045268011 x2_input_acceleration
   - 0.1538699756292546 x3_input_acceleration >= -0.0624047197465499
 Relaxed_CLF_constraint: 0.0216726466453253 x1_input_acceleration
   + 0.3086571295256375 x2_input_acceleration
   - 0.0710581910922151 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0979636305246547 x1_input_acceleration
   + 0.3011320452685179 x2_input_acceleration
   - 0.1549221210209688 x3_input_acceleration >= -0.0210695612937019
 CBF_Constraint_h_left: 0.2820363694753453 x1_input_acceleration
   + 0.3011320452685179 x2_input_acceleration
   - 0.1549221210209688 x3_input_acceleration >= -0.0385564714940175
 CBF_Constraint_h_left: - 0.4179636305246547 x1_input_acceleration
   + 0.3011320452685179 x2_input_acceleration
   - 0.1549221210209688 x3_input_acceleration >= -0.0623437421776466
 Relaxed_CLF_constraint: 0.0209254398341081 x1_input_acceleration
   + 0.307157129525891 x2_input_acceleration
   - 0.0715842637880723 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.0994434953278157 x1_input_acceleration
   + 0.2981320452730647 x2_input_acceleration
   - 0.1559812188967974 x3_input_acceleration >= -0.0207754664575612
 CBF_Constraint_h_left: 0.2805565046721843 x1_input_acceleration
   + 0.2981320452730647 x2_input_acceleration
   - 0.1559812188967974 x3_input_acceleration >= -0.0379812023452762
 CBF_Constraint_h_left: - 0.4194434953278157 x1_input_acceleration
   + 0.2981320452730647 x2_input_acceleration
   - 0.1559812188967974 x3_input_acceleration >= -0.0622864257100117
 Relaxed_CLF_constraint: 0.0201855074325276 x1_input_acceleration
   + 0.3056571295281644 x2_input_acceleration
   - 0.0721138127259865 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1008913105657948 x1_input_acceleration
   + 0.2951275153205284 x2_input_acceleration
   - 0.1570511456798734 x3_input_acceleration >= -0.0204860923015783
 CBF_Constraint_h_left: 0.2791086894342052 x1_input_acceleration
   + 0.2951275153205284 x2_input_acceleration
   - 0.1570511456798734 x3_input_acceleration >= -0.0374167432940773
 CBF_Constraint_h_left: - 0.4209094303778065 x1_input_acceleration
   + 0.2951320452735313 x2_input_acceleration
   - 0.1570469733547391 x3_input_acceleration >= -0.0622328561420478
 Relaxed_CLF_constraint: 0.0194525399075322 x1_input_acceleration
   + 0.3041571295283977 x2_input_acceleration
   - 0.0726466899549574 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1023440947276195 x1_input_acceleration
   + 0.2921275153277222 x2_input_acceleration
   - 0.1581231846768471 x3_input_acceleration >= -0.0202039351173782
 CBF_Constraint_h_left: 0.2776559052723805 x1_input_acceleration
   + 0.2921275153277222 x2_input_acceleration
   - 0.1581231846768471 x3_input_acceleration >= -0.0368585571191305
 CBF_Constraint_h_left: - 0.4223440947276195 x1_input_acceleration
   + 0.2921275153277222 x2_input_acceleration
   - 0.1581231846768471 x3_input_acceleration >= -0.0621789902737973
 Relaxed_CLF_constraint: 0.0187261478266199 x1_input_acceleration
   + 0.3026571295319946 x2_input_acceleration
   - 0.0731827094534443 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1035195346616798 x1_input_acceleration
   + 0.2898688778996016 x2_input_acceleration
   - 0.157999543470976 x3_input_acceleration >= -0.0199260290420954
 CBF_Constraint_h_left: 0.2764804653383202 x1_input_acceleration
   + 0.2898688778996016 x2_input_acceleration
   - 0.157999543470976 x3_input_acceleration >= -0.0363573174563763
 CBF_Constraint_h_left: - 0.4237841792844825 x1_input_acceleration
   + 0.2891275153278853 x2_input_acceleration
   - 0.1592012923189375 x3_input_acceleration >= -0.0621332005518797
 Relaxed_CLF_constraint: 0.0180061055481884 x1_input_acceleration
   + 0.3011571295320761 x2_input_acceleration
   - 0.0737217632744895 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.104940579034783 x1_input_acceleration
   + 0.2868688779034078 x2_input_acceleration
   - 0.1590867499113363 x3_input_acceleration >= -0.019653718058767
 CBF_Constraint_h_left: 0.275059420965217 x1_input_acceleration
   + 0.2868688779034078 x2_input_acceleration
   - 0.1590867499113363 x3_input_acceleration >= -0.0358150080421582
 CBF_Constraint_h_left: - 0.424940579034783 x1_input_acceleration
   + 0.2868688779034078 x2_input_acceleration
   - 0.1590867499113363 x3_input_acceleration >= -0.0620442107043323
 Relaxed_CLF_constraint: 0.0172955833616368 x1_input_acceleration
   + 0.2996571295339792 x2_input_acceleration
   - 0.0742653664946696 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1063500366504711 x1_input_acceleration
   + 0.2838688779039341 x2_input_acceleration
   - 0.1601794932620544 x3_input_acceleration >= -0.0193873350499209
 CBF_Constraint_h_left: 0.2736499633495289 x1_input_acceleration
   + 0.2838688779039341 x2_input_acceleration
   - 0.1601794932620544 x3_input_acceleration >= -0.0352808280863314
 CBF_Constraint_h_left: - 0.4263500366504711 x1_input_acceleration
   + 0.2838688779039341 x2_input_acceleration
   - 0.1601794932620544 x3_input_acceleration >= -0.0620033409139963
 Relaxed_CLF_constraint: 0.0165908545537927 x1_input_acceleration
   + 0.2981571295342424 x2_input_acceleration
   - 0.0748117381700286 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1077485154121461 x1_input_acceleration
   + 0.2808688779042908 x2_input_acceleration
   - 0.1612774830831498 x3_input_acceleration >= -0.0191268739245932
 CBF_Constraint_h_left: 0.2722514845878539 x1_input_acceleration
   + 0.2808688779042908 x2_input_acceleration
   - 0.1612774830831498 x3_input_acceleration >= -0.0347546559962854
 CBF_Constraint_h_left: - 0.4277485154121461 x1_input_acceleration
   + 0.2808688779042908 x2_input_acceleration
   - 0.1612774830831498 x3_input_acceleration >= -0.0619666363905365
 Relaxed_CLF_constraint: 0.0158916151729552 x1_input_acceleration
   + 0.2966571295344207 x2_input_acceleration
   - 0.0753607330805764 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1093879379119972 x1_input_acceleration
   + 0.2771325221234289 x2_input_acceleration
   - 0.1635799555316318 x3_input_acceleration >= -0.0188816394077156
 CBF_Constraint_h_left: 0.2706120620880028 x1_input_acceleration
   + 0.2771325221234289 x2_input_acceleration
   - 0.1635799555316318 x3_input_acceleration >= -0.0341979312044361
 CBF_Constraint_h_left: - 0.4293879379119973 x1_input_acceleration
   + 0.2771325221234289 x2_input_acceleration
   - 0.1635799555316318 x3_input_acceleration >= -0.0619837094736351
 Relaxed_CLF_constraint: 0.0151975505142161 x1_input_acceleration
   + 0.2951571295347674 x2_input_acceleration
   - 0.0759122273690873 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1107741350991121 x1_input_acceleration
   + 0.2741325221319981 x2_input_acceleration
   - 0.1646838885658194 x3_input_acceleration >= -0.0186350829626415
 CBF_Constraint_h_left: 0.2692258649008879 x1_input_acceleration
   + 0.2741325221319981 x2_input_acceleration
   - 0.1646838885658194 x3_input_acceleration >= -0.0336879972938102
 CBF_Constraint_h_left: - 0.4307741350991121 x1_input_acceleration
   + 0.2741325221319981 x2_input_acceleration
   - 0.1646838885658194 x3_input_acceleration >= -0.0619589445784994
 Relaxed_CLF_constraint: 0.0145044519206587 x1_input_acceleration
   + 0.293657129539052 x2_input_acceleration
   - 0.0764641938861812 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1121513028393217 x1_input_acceleration
   + 0.2711325221322383 x2_input_acceleration
   - 0.1657922351395985 x3_input_acceleration >= -0.0183944561297325
 CBF_Constraint_h_left: 0.2678486971606783 x1_input_acceleration
   + 0.2711325221322383 x2_input_acceleration
   - 0.1657922351395985 x3_input_acceleration >= -0.0331857085902613
 CBF_Constraint_h_left: - 0.4321513028393217 x1_input_acceleration
   + 0.2711325221322383 x2_input_acceleration
   - 0.1657922351395985 x3_input_acceleration >= -0.0619386645840239
 Relaxed_CLF_constraint: 0.0138158680505539 x1_input_acceleration
   + 0.2921571295391721 x2_input_acceleration
   - 0.0770183671730707 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1135200333691126 x1_input_acceleration
   + 0.2681325221335733 x2_input_acceleration
   - 0.1669047058071532 x3_input_acceleration >= -0.018159757055602
 CBF_Constraint_h_left: 0.2664799666308874 x1_input_acceleration
   + 0.2681325221335733 x2_input_acceleration
   - 0.1669047058071532 x3_input_acceleration >= -0.0326909507154706
 CBF_Constraint_h_left: - 0.4335200333691127 x1_input_acceleration
   + 0.2681325221335733 x2_input_acceleration
   - 0.1669047058071532 x3_input_acceleration >= -0.06192296239466
 Relaxed_CLF_constraint: 0.0131315027856584 x1_input_acceleration
   + 0.2906571295398396 x2_input_acceleration
   - 0.077574602506848 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1149009427042221 x1_input_acceleration
   + 0.2651151175793044 x2_input_acceleration
   - 0.1680417551680223 x3_input_acceleration >= -0.0179315709208392
 CBF_Constraint_h_left: 0.2651190844564225 x1_input_acceleration
   + 0.2651325221355788 x2_input_acceleration
   - 0.1680210127516405 x3_input_acceleration >= -0.0322036109907679
 CBF_Constraint_h_left: - 0.4348809155435776 x1_input_acceleration
   + 0.2651325221355788 x2_input_acceleration
   - 0.1680210127516405 x3_input_acceleration >= -0.0619119314310201
 Relaxed_CLF_constraint: 0.012451061698426 x1_input_acceleration
   + 0.2891571295408423 x2_input_acceleration
   - 0.0781327559790917 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1162545618949902 x1_input_acceleration
   + 0.262115117580096 x2_input_acceleration
   - 0.1691616122299248 x3_input_acceleration >= -0.0177087772694128
 CBF_Constraint_h_left: 0.2637454381050098 x1_input_acceleration
   + 0.262115117580096 x2_input_acceleration
   - 0.1691616122299248 x3_input_acceleration >= -0.0317204105093646
 CBF_Constraint_h_left: - 0.4362345347343456 x1_input_acceleration
   + 0.2621325221363704 x2_input_acceleration
   - 0.169140869813543 x3_input_acceleration >= -0.0619056655744369
 Relaxed_CLF_constraint: 0.0117742521030419 x1_input_acceleration
   + 0.2876571295412381 x2_input_acceleration
   - 0.078692684510043 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.117581017475608 x1_input_acceleration
   + 0.2591197667432085 x2_input_acceleration
   - 0.1702975991004959 x3_input_acceleration >= -0.0174924053617618
 CBF_Constraint_h_left: 0.262418982524392 x1_input_acceleration
   + 0.2591197667432085 x2_input_acceleration
   - 0.1702975991004959 x3_input_acceleration >= -0.0312520120413963
 CBF_Constraint_h_left: - 0.437581017475608 x1_input_acceleration
   + 0.2591197667432085 x2_input_acceleration
   - 0.1702975991004959 x3_input_acceleration >= -0.0619053681578591
 Relaxed_CLF_constraint: 0.0111007723138317 x1_input_acceleration
   + 0.2861571295416481 x2_input_acceleration
   - 0.0792542406436927 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1189221812482964 x1_input_acceleration
   + 0.2561197667434985 x2_input_acceleration
   - 0.1714235443447308 x3_input_acceleration >= -0.0172814629163267
 CBF_Constraint_h_left: 0.2610778187517036 x1_input_acceleration
   + 0.2561197667434985 x2_input_acceleration
   - 0.1714235443447308 x3_input_acceleration >= -0.0307862484791503
 CBF_Constraint_h_left: - 0.4389221812482964 x1_input_acceleration
   + 0.2561197667434985 x2_input_acceleration
   - 0.1714235443447308 x3_input_acceleration >= -0.0619090119160541
 Relaxed_CLF_constraint: 0.0104301904274875 x1_input_acceleration
   + 0.2846571295417931 x2_input_acceleration
   - 0.0798172132658102 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1202578170284451 x1_input_acceleration
   + 0.2531197667462611 x2_input_acceleration
   - 0.1725521916555616 x3_input_acceleration >= -0.0170764544298166
 CBF_Constraint_h_left: 0.2597421829715549 x1_input_acceleration
   + 0.2531197667462611 x2_input_acceleration
   - 0.1725521916555616 x3_input_acceleration >= -0.030327469194412
 CBF_Constraint_h_left: - 0.4402578170284451 x1_input_acceleration
   + 0.2531197667462611 x2_input_acceleration
   - 0.1725521916555616 x3_input_acceleration >= -0.0619177051543678
 Relaxed_CLF_constraint: 0.00976237253741313 x1_input_acceleration
   + 0.2831571295431745 x2_input_acceleration
   - 0.0803815369212256 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1213310038130485 x1_input_acceleration
   + 0.2508607716964645 x2_input_acceleration
   - 0.172434186994401 x3_input_acceleration >= -0.0168464720267119
 CBF_Constraint_h_left: 0.2584115041215218 x1_input_acceleration
   + 0.2501197667526168 x2_input_acceleration
   - 0.1736832619303141 x3_input_acceleration >= -0.0298755696643712
 CBF_Constraint_h_left: - 0.4415884958784782 x1_input_acceleration
   + 0.2501197667526168 x2_input_acceleration
   - 0.1736832619303141 x3_input_acceleration >= -0.0619315432218386
 Relaxed_CLF_constraint: 0.00909703311239657 x1_input_acceleration
   + 0.2816571295463523 x2_input_acceleration
   - 0.0809470720586019 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1226572931229392 x1_input_acceleration
   + 0.2478607716969091 x2_input_acceleration
   - 0.1735674028483253 x3_input_acceleration >= -0.0166513542584867
 CBF_Constraint_h_left: 0.2570852148116312 x1_input_acceleration
   + 0.2471197667530615 x2_input_acceleration
   - 0.1748164777842383 x3_input_acceleration >= -0.0294304469249293
 CBF_Constraint_h_left: - 0.4429147851883689 x1_input_acceleration
   + 0.2471197667530615 x2_input_acceleration
   - 0.1748164777842383 x3_input_acceleration >= -0.0619506217408584
 Relaxed_CLF_constraint: 0.00843388845745124 x1_input_acceleration
   + 0.2801571295465746 x2_input_acceleration
   - 0.081513679985564 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1239797566660486 x1_input_acceleration
   + 0.2448607716989308 x2_input_acceleration
   - 0.1747024887330991 x3_input_acceleration >= -0.0164621842873768
 CBF_Constraint_h_left: 0.2560202433339514 x1_input_acceleration
   + 0.2448607716989308 x2_input_acceleration
   - 0.1747024887330991 x3_input_acceleration >= -0.0290060305208275
 CBF_Constraint_h_left: - 0.4439797566660486 x1_input_acceleration
   + 0.2448607716989308 x2_input_acceleration
   - 0.1747024887330991 x3_input_acceleration >= -0.0618989453539446
 Relaxed_CLF_constraint: 0.00777265668589652 x1_input_acceleration
   + 0.2786571295475855 x2_input_acceleration
   - 0.0820812229279509 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1252885227347877 x1_input_acceleration
   + 0.241860771715027 x2_input_acceleration
   - 0.1758442699268442 x3_input_acceleration >= -0.0162787635224397
 CBF_Constraint_h_left: 0.2547114772652123 x1_input_acceleration
   + 0.241860771715027 x2_input_acceleration
   - 0.1758442699268442 x3_input_acceleration >= -0.02857394420283
 CBF_Constraint_h_left: - 0.4452885227347877 x1_input_acceleration
   + 0.241860771715027 x2_input_acceleration
   - 0.1758442699268442 x3_input_acceleration >= -0.0619249271600057
 Relaxed_CLF_constraint: 0.00711827365152698 x1_input_acceleration
   + 0.2771571295556335 x2_input_acceleration
   - 0.0826521135248234 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.126603571616211 x1_input_acceleration
   + 0.2388749576344066 x2_input_acceleration
   - 0.1770119656347493 x3_input_acceleration >= -0.0161057364271746
 CBF_Constraint_h_left: 0.2534054882897949 x1_input_acceleration
   + 0.23886077172895 x2_input_acceleration
   - 0.1769874085211019 x3_input_acceleration >= -0.0281483381353386
 CBF_Constraint_h_left: - 0.4465945117102051 x1_input_acceleration
   + 0.23886077172895 x2_input_acceleration
   - 0.1769874085211019 x3_input_acceleration >= -0.0619564172339104
 Relaxed_CLF_constraint: 0.00646527916381825 x1_input_acceleration
   + 0.2756571295625951 x2_input_acceleration
   - 0.0832236828219523 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1279073370950144 x1_input_acceleration
   + 0.235874957636776 x2_input_acceleration
   - 0.1781561910937901 x3_input_acceleration >= -0.0159342277369839
 CBF_Constraint_h_left: 0.2521017228109915 x1_input_acceleration
   + 0.2358607717313195 x2_input_acceleration
   - 0.1781316339801426 x3_input_acceleration >= -0.0277291153275998
 CBF_Constraint_h_left: - 0.4478982771890085 x1_input_acceleration
   + 0.2358607717313195 x2_input_acceleration
   - 0.1781316339801426 x3_input_acceleration >= -0.0619935123437528
 Relaxed_CLF_constraint: 0.00581339642441656 x1_input_acceleration
   + 0.2741571295637798 x2_input_acceleration
   - 0.0837957955514727 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1292094290034899 x1_input_acceleration
   + 0.2328749576371753 x2_input_acceleration
   - 0.1793012345899836 x3_input_acceleration >= -0.0157686887908541
 CBF_Constraint_h_left: 0.2507996309025159 x1_input_acceleration
   + 0.2328607717317187 x2_input_acceleration
   - 0.1792766774763362 x3_input_acceleration >= -0.027316180239821
 CBF_Constraint_h_left: - 0.4492003690974841 x1_input_acceleration
   + 0.2328607717317187 x2_input_acceleration
   - 0.1792766774763362 x3_input_acceleration >= -0.0620363094239404
 Relaxed_CLF_constraint: 0.00516235047017879 x1_input_acceleration
   + 0.2726571295639794 x2_input_acceleration
   - 0.0843683172995694 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1305103936522157 x1_input_acceleration
   + 0.2298749576779812 x2_input_acceleration
   - 0.1804468290318806 x3_input_acceleration >= -0.0156091292815927
 CBF_Constraint_h_left: 0.2494896063477843 x1_input_acceleration
   + 0.2298749576779812 x2_input_acceleration
   - 0.1804468290318806 x3_input_acceleration >= -0.0269121544876717
 CBF_Constraint_h_left: - 0.4505013337462099 x1_input_acceleration
   + 0.2298607717725246 x2_input_acceleration
   - 0.1804222719182332 x3_input_acceleration >= -0.0620849055777779
 Relaxed_CLF_constraint: 0.00451186814581589 x1_input_acceleration
   + 0.2711571295843824 x2_input_acceleration
   - 0.0849411145205179 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1318110073837047 x1_input_acceleration
   + 0.2268749576790272 x2_input_acceleration
   - 0.1815925950579163 x3_input_acceleration >= -0.0154555646673089
 CBF_Constraint_h_left: 0.2481889926162953 x1_input_acceleration
   + 0.2268749576790272 x2_input_acceleration
   - 0.1815925950579163 x3_input_acceleration >= -0.0265114732644051
 CBF_Constraint_h_left: - 0.4518110073837047 x1_input_acceleration
   + 0.2268749576790272 x2_input_acceleration
   - 0.1815925950579163 x3_input_acceleration >= -0.0621453258487017
 Relaxed_CLF_constraint: 0.0038615612800714 x1_input_acceleration
   + 0.2696571295849054 x2_input_acceleration
   - 0.0855139975335357 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1333676350021955 x1_input_acceleration
   + 0.2231297804107233 x2_input_acceleration
   - 0.1839552728110497 x3_input_acceleration >= -0.015353341841801
 CBF_Constraint_h_left: 0.2466323649978046 x1_input_acceleration
   + 0.2231297804107233 x2_input_acceleration
   - 0.1839552728110497 x3_input_acceleration >= -0.0261134911913839
 CBF_Constraint_h_left: - 0.4533614361191388 x1_input_acceleration
   + 0.223130376457171 x2_input_acceleration
   - 0.1839340535575097 x3_input_acceleration >= -0.0622888731790059
 Relaxed_CLF_constraint: 0.00321127824780357 x1_input_acceleration
   + 0.2681571295852444 x2_input_acceleration
   - 0.0860868921962353 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1346682010680145 x1_input_acceleration
   + 0.2201297804141107 x2_input_acceleration
   - 0.1851010621440423 x3_input_acceleration >= -0.0152137619527281
 CBF_Constraint_h_left: 0.2453317989319855 x1_input_acceleration
   + 0.2201297804141107 x2_input_acceleration
   - 0.1851010621440423 x3_input_acceleration >= -0.0257268037498053
 CBF_Constraint_h_left: - 0.4546682010680145 x1_input_acceleration
   + 0.2201297804141107 x2_input_acceleration
   - 0.1851010621440423 x3_input_acceleration >= -0.0623606741236104
 Relaxed_CLF_constraint: 0.00256099521489403 x1_input_acceleration
   + 0.2666571295869381 x2_input_acceleration
   - 0.0866597868627316 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.135968767133321 x1_input_acceleration
   + 0.2171297804144016 x2_input_acceleration
   - 0.1862468514689165 x3_input_acceleration >= -0.0150801842151615
 CBF_Constraint_h_left: 0.244031232866679 x1_input_acceleration
   + 0.2171297804144016 x2_input_acceleration
   - 0.1862468514689165 x3_input_acceleration >= -0.0253461184598305
 CBF_Constraint_h_left: - 0.455968767133321 x1_input_acceleration
   + 0.2171297804144016 x2_input_acceleration
   - 0.1862468514689165 x3_input_acceleration >= -0.0624351869564929
 Relaxed_CLF_constraint: 0.00191071218224079 x1_input_acceleration
   + 0.2651571295870835 x2_input_acceleration
   - 0.0872326815251687 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1369980128543444 x1_input_acceleration
   + 0.2148630367552192 x2_input_acceleration
   - 0.1861641890648586 x3_input_acceleration >= -0.0148979213449726
 CBF_Constraint_h_left: 0.2430019871456556 x1_input_acceleration
   + 0.2148630367552192 x2_input_acceleration
   - 0.1861641890648586 x3_input_acceleration >= -0.0249682989026471
 CBF_Constraint_h_left: - 0.457269333197362 x1_input_acceleration
   + 0.2141297804151924 x2_input_acceleration
   - 0.1873926407936794 x3_input_acceleration >= -0.0625157019417619
 Relaxed_CLF_constraint: 0.00126042915022029 x1_input_acceleration
   + 0.2636571295874789 x2_input_acceleration
   - 0.0878055761875501 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1382985788972886 x1_input_acceleration
   + 0.2118630367649451 x2_input_acceleration
   - 0.1873099783794326 x3_input_acceleration >= -0.0147743678181944
 CBF_Constraint_h_left: 0.2417014211027114 x1_input_acceleration
   + 0.2118630367649451 x2_input_acceleration
   - 0.1873099783794326 x3_input_acceleration >= -0.0245976378277095
 CBF_Constraint_h_left: - 0.4585698992403062 x1_input_acceleration
   + 0.2111297804249183 x2_input_acceleration
   - 0.1885384301082534 x3_input_acceleration >= -0.0626022190748059
 Relaxed_CLF_constraint: 6.1014612874818464e-04 x1_input_acceleration
   + 0.2621571295923419 x2_input_acceleration
   - 0.0883784708448371 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.139603913334092 x1_input_acceleration
   + 0.2088775802991139 x2_input_acceleration
   - 0.1884412241709223 x3_input_acceleration >= -0.0146572977842103
 CBF_Constraint_h_left: 0.24040085503749 x1_input_acceleration
   + 0.2088630367657887 x2_input_acceleration
   - 0.1884557677042475 x3_input_acceleration >= -0.0242329789026952
 CBF_Constraint_h_left: - 0.45959914496251 x1_input_acceleration
   + 0.2088630367657887 x2_input_acceleration
   - 0.1884557677042475 x3_input_acceleration >= -0.0625926796395737
 Relaxed_CLF_constraint: - 4.01369e-05 x1_input_acceleration
   + 0.2606571295927637 x2_input_acceleration
   - 0.0889513655072446 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1409044793840104 x1_input_acceleration
   + 0.2058775803019566 x2_input_acceleration
   - 0.1895870135014258 x3_input_acceleration >= -0.0145457215174643
 CBF_Constraint_h_left: 0.2390955206159897 x1_input_acceleration
   + 0.2058775803019566 x2_input_acceleration
   - 0.1895870135014258 x3_input_acceleration >= -0.0238738704345024
 CBF_Constraint_h_left: - 0.4609044793840104 x1_input_acceleration
   + 0.2058775803019566 x2_input_acceleration
   - 0.1895870135014258 x3_input_acceleration >= -0.062690438218906
 Relaxed_CLF_constraint: - 0.00069042 x1_input_acceleration
   + 0.259157129594185 x2_input_acceleration
   - 0.0895242601724963 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1424835181805708 x1_input_acceleration
   + 0.2021244161299912 x2_input_acceleration
   - 0.1919650691613111 x3_input_acceleration >= -0.0145016050817775
 CBF_Constraint_h_left: 0.2375164818194292 x1_input_acceleration
   + 0.2021244161299912 x2_input_acceleration
   - 0.1919650691613111 x3_input_acceleration >= -0.023529736627469
 CBF_Constraint_h_left: - 0.4624835181805708 x1_input_acceleration
   + 0.2021244161299912 x2_input_acceleration
   - 0.1919650691613111 x3_input_acceleration >= -0.0628989679906688
 Relaxed_CLF_constraint: - 0.00134070287690657 x1_input_acceleration
   + 0.2576571296538933 x2_input_acceleration
   - 0.0900971547893958 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1435141943679734 x1_input_acceleration
   + 0.1998707855015439 x2_input_acceleration
   - 0.1918855062345785 x3_input_acceleration >= -0.0143411755962483
 CBF_Constraint_h_left: 0.2362159158004836 x1_input_acceleration
   + 0.1991244161396665 x2_input_acceleration
   - 0.193110858521871 x3_input_acceleration >= -0.0231850739148696
 CBF_Constraint_h_left: - 0.4637840841995164 x1_input_acceleration
   + 0.1991244161396665 x2_input_acceleration
   - 0.193110858521871 x3_input_acceleration >= -0.0630095033847003
 Relaxed_CLF_constraint: - 0.00199098588637936 x1_input_acceleration
   + 0.2561571296587309 x2_input_acceleration
   - 0.0906700494696757 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1448147603264544 x1_input_acceleration
   + 0.1968707855545817 x2_input_acceleration
   - 0.1930312955003151 x3_input_acceleration >= -0.0142476255139541
 CBF_Constraint_h_left: 0.2351852396735456 x1_input_acceleration
   + 0.1968707855545817 x2_input_acceleration
   - 0.1930312955003151 x3_input_acceleration >= -0.0228328210519278
 CBF_Constraint_h_left: - 0.4648147603264544 x1_input_acceleration
   + 0.1968707855545817 x2_input_acceleration
   - 0.1930312955003151 x3_input_acceleration >= -0.0630179871661868
 Relaxed_CLF_constraint: - 0.00264126886561986 x1_input_acceleration
   + 0.2546571296852498 x2_input_acceleration
   - 0.091242944102544 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1463756794720793 x1_input_acceleration
   + 0.1931228664738317 x2_input_acceleration
   - 0.1954172190661149 x3_input_acceleration >= -0.0142275426508791
 CBF_Constraint_h_left: 0.2336243205279207 x1_input_acceleration
   + 0.1931228664738317 x2_input_acceleration
   - 0.1954172190661149 x3_input_acceleration >= -0.022516163551184
 CBF_Constraint_h_left: - 0.4663756794720793 x1_input_acceleration
   + 0.1931228664738317 x2_input_acceleration
   - 0.1954172190661149 x3_input_acceleration >= -0.0632476513664118
 Relaxed_CLF_constraint: - 0.00329155189424285 x1_input_acceleration
   + 0.2531571296861956 x2_input_acceleration
   - 0.0918158387658455 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1476762454267708 x1_input_acceleration
   + 0.1901228665283339 x2_input_acceleration
   - 0.1965630083259634 x3_input_acceleration >= -0.0141479985206128
 CBF_Constraint_h_left: 0.2323237545732292 x1_input_acceleration
   + 0.1901228665283339 x2_input_acceleration
   - 0.1965630083259634 x3_input_acceleration >= -0.0221895118895264
 CBF_Constraint_h_left: - 0.4676762454267708 x1_input_acceleration
   + 0.1901228665283339 x2_input_acceleration
   - 0.1965630083259634 x3_input_acceleration >= -0.0633761977888961
 Relaxed_CLF_constraint: - 0.00394183487158858 x1_input_acceleration
   + 0.2516571297134467 x2_input_acceleration
   - 0.0923887333957697 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1489768109546948 x1_input_acceleration
   + 0.1871228669865553 x2_input_acceleration
   - 0.1977087968766077 x3_input_acceleration >= -0.0140744564784737
 CBF_Constraint_h_left: 0.2310231890453052 x1_input_acceleration
   + 0.1871228669865553 x2_input_acceleration
   - 0.1977087968766077 x3_input_acceleration >= -0.0218688623970816
 CBF_Constraint_h_left: - 0.4689768109546948 x1_input_acceleration
   + 0.1871228669865553 x2_input_acceleration
   - 0.1977087968766077 x3_input_acceleration >= -0.0635107462312249
 Relaxed_CLF_constraint: - 0.00459211763555062 x1_input_acceleration
   + 0.2501571299425574 x2_input_acceleration
   - 0.0929616276710919 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.150004626166139 x1_input_acceleration
   + 0.184864348767829 x2_input_acceleration
   - 0.1976016965677998 x3_input_acceleration >= -0.0139306614507674
 CBF_Constraint_h_left: 0.229995373833861 x1_input_acceleration
   + 0.184864348767829 x2_input_acceleration
   - 0.1976016965677998 x3_input_acceleration >= -0.0215297824792009
 CBF_Constraint_h_left: - 0.470004626166139 x1_input_acceleration
   + 0.184864348767829 x2_input_acceleration
   - 0.1976016965677998 x3_input_acceleration >= -0.0635314016373496
 Relaxed_CLF_constraint: - 0.0052424006685188 x1_input_acceleration
   + 0.2486571299426913 x2_input_acceleration
   - 0.0935345223332773 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1513051922307829 x1_input_acceleration
   + 0.1818643487699877 x2_input_acceleration
   - 0.1987474858886928 x3_input_acceleration >= -0.0138671164241505
 CBF_Constraint_h_left: 0.2286948077692171 x1_input_acceleration
   + 0.1818643487699877 x2_input_acceleration
   - 0.1987474858886928 x3_input_acceleration >= -0.0212191299003018
 CBF_Constraint_h_left: - 0.4713051922307829 x1_input_acceleration
   + 0.1818643487699877 x2_input_acceleration
   - 0.1987474858886928 x3_input_acceleration >= -0.0636759471810758
 Relaxed_CLF_constraint: - 0.00589268370084073 x1_input_acceleration
   + 0.2471571299437706 x2_input_acceleration
   - 0.0941074169937238 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1526129108406033 x1_input_acceleration
   + 0.1788555272846879 x2_input_acceleration
   - 0.1999315413973393 x3_input_acceleration >= -0.0138131553602604
 CBF_Constraint_h_left: 0.2273942417167698 x1_input_acceleration
   + 0.1788643487721147 x2_input_acceleration
   - 0.1998932752153935 x3_input_acceleration >= -0.0209144794759886
 CBF_Constraint_h_left: - 0.4726057582832303 x1_input_acceleration
   + 0.1788643487721147 x2_input_acceleration
   - 0.1998932752153935 x3_input_acceleration >= -0.0638264948751192
 Relaxed_CLF_constraint: - 0.00654296672706442 x1_input_acceleration
   + 0.2456571299448341 x2_input_acceleration
   - 0.0946803116570742 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1539134767742502 x1_input_acceleration
   + 0.1758555278107733 x2_input_acceleration
   - 0.2010773299223894 x3_input_acceleration >= -0.0137616544007652
 CBF_Constraint_h_left: 0.2260865232257498 x1_input_acceleration
   + 0.1758555278107733 x2_input_acceleration
   - 0.2010773299223894 x3_input_acceleration >= -0.0206180938136576
 CBF_Constraint_h_left: - 0.4739063242168772 x1_input_acceleration
   + 0.1758643492982001 x2_input_acceleration
   - 0.2010390637404436 x3_input_acceleration >= -0.0639830446591163
 Relaxed_CLF_constraint: - 0.00719324969388788 x1_input_acceleration
   + 0.2441571302078768 x2_input_acceleration
   - 0.0952532059195992 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1554829789723651 x1_input_acceleration
   + 0.1721207217705466 x2_input_acceleration
   - 0.2034217686165515 x3_input_acceleration >= -0.0137952288900052
 CBF_Constraint_h_left: 0.2247859571848614 x1_input_acceleration
   + 0.1728555278313376 x2_input_acceleration
   - 0.2022231192101184 x3_input_acceleration >= -0.0203254874981086
 CBF_Constraint_h_left: - 0.4752140428151386 x1_input_acceleration
   + 0.1728555278313376 x2_input_acceleration
   - 0.2022231192101184 x3_input_acceleration >= -0.0641504024834071
 Relaxed_CLF_constraint: - 0.00784353271433205 x1_input_acceleration
   + 0.242657130218159 x2_input_acceleration
   - 0.0958261005634637 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1567835450350675 x1_input_acceleration
   + 0.1691207217730174 x2_input_acceleration
   - 0.2045675579369545 x3_input_acceleration >= -0.0137576960717697
 CBF_Constraint_h_left: 0.223485391122159 x1_input_acceleration
   + 0.1698555278338084 x2_input_acceleration
   - 0.2033689085305214 x3_input_acceleration >= -0.0200388833344054
 CBF_Constraint_h_left: - 0.476514608877841 x1_input_acceleration
   + 0.1698555278338084 x2_input_acceleration
   - 0.2033689085305214 x3_input_acceleration >= -0.0643189964416498
 Relaxed_CLF_constraint: - 0.00849381574568325 x1_input_acceleration
   + 0.2411571302193944 x2_input_acceleration
   - 0.0963989952236652 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1580841105137604 x1_input_acceleration
   + 0.1661207223105184 x2_input_acceleration
   - 0.2057133461412174 x3_input_acceleration >= -0.0137261652896279
 CBF_Constraint_h_left: 0.2219158894862396 x1_input_acceleration
   + 0.1661207223105184 x2_input_acceleration
   - 0.2057133461412174 x3_input_acceleration >= -0.0197901842920134
 CBF_Constraint_h_left: - 0.4780841105137604 x1_input_acceleration
   + 0.1661207223105184 x2_input_acceleration
   - 0.2057133461412174 x3_input_acceleration >= -0.0646196229718295
 Relaxed_CLF_constraint: - 0.0091440984850297 x1_input_acceleration
   + 0.2396571304881449 x2_input_acceleration
   - 0.0969718893257966 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1591104659685734 x1_input_acceleration
   + 0.1638664145467345 x2_input_acceleration
   - 0.2056500281495675 x3_input_acceleration >= -0.0136150690687642
 CBF_Constraint_h_left: 0.2208895340314266 x1_input_acceleration
   + 0.1638664145467345 x2_input_acceleration
   - 0.2056500281495675 x3_input_acceleration >= -0.0194840805347352
 CBF_Constraint_h_left: - 0.4791104659685734 x1_input_acceleration
   + 0.1638664145467345 x2_input_acceleration
   - 0.2056500281495675 x3_input_acceleration >= -0.0646727436237359
 Relaxed_CLF_constraint: - 0.00979436689541957 x1_input_acceleration
   + 0.2381571495531828 x2_input_acceleration
   - 0.0975447406372837 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1606898540405403 x1_input_acceleration
   + 0.1601410735912807 x2_input_acceleration
   - 0.207977335783672 x3_input_acceleration >= -0.0136802412105531
 CBF_Constraint_h_left: 0.2195924335571159 x1_input_acceleration
   + 0.1608709920712001 x2_input_acceleration
   - 0.2067853620974537 x3_input_acceleration >= -0.0192151247358209
 CBF_Constraint_h_left: - 0.4804075664428841 x1_input_acceleration
   + 0.1608709920712001 x2_input_acceleration
   - 0.2067853620974537 x3_input_acceleration >= -0.0648577729908303
 Relaxed_CLF_constraint: - 0.0104429171325749 x1_input_acceleration
   + 0.2366594383154157 x2_input_acceleration
   - 0.0981124076112268 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1619362518229902 x1_input_acceleration
   + 0.157204376318385 x2_input_acceleration
   - 0.2089691532198145 x3_input_acceleration >= -0.0136511681463844
 CBF_Constraint_h_left: 0.2180637481770098 x1_input_acceleration
   + 0.157204376318385 x2_input_acceleration
   - 0.2089691532198145 x3_input_acceleration >= -0.0189832803000163
 CBF_Constraint_h_left: - 0.4819362518229902 x1_input_acceleration
   + 0.157204376318385 x2_input_acceleration
   - 0.2089691532198145 x3_input_acceleration >= -0.0651609684380628
 Relaxed_CLF_constraint: - 0.0110689770467491 x1_input_acceleration
   + 0.2351924605857977 x2_input_acceleration
   - 0.0986109389336682 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1631326920143831 x1_input_acceleration
   + 0.1543392087313249 x2_input_acceleration
   - 0.2098141443632852 x3_input_acceleration >= -0.0136137104326421
 CBF_Constraint_h_left: 0.2168673079856169 x1_input_acceleration
   + 0.1543392087313249 x2_input_acceleration
   - 0.2098141443632852 x3_input_acceleration >= -0.0187184989499093
 CBF_Constraint_h_left: - 0.4831326920143831 x1_input_acceleration
   + 0.1543392087313249 x2_input_acceleration
   - 0.2098141443632852 x3_input_acceleration >= -0.0653149411549434
 Relaxed_CLF_constraint: - 0.0116671971424455 x1_input_acceleration
   + 0.2337598767922676 x2_input_acceleration
   - 0.0990334345054036 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1643147134089883 x1_input_acceleration
   + 0.1514514028544003 x2_input_acceleration
   - 0.2106699551375309 x3_input_acceleration >= -0.0135796706167233
 CBF_Constraint_h_left: 0.2156738424992148 x1_input_acceleration
   + 0.1514776288981015 x2_input_acceleration
   - 0.2106512392790714 x3_input_acceleration >= -0.0184586557511939
 CBF_Constraint_h_left: - 0.4843261575007852 x1_input_acceleration
   + 0.1514776288981015 x2_input_acceleration
   - 0.2106512392790714 x3_input_acceleration >= -0.0654728108764687
 Relaxed_CLF_constraint: - 0.0122639298856466 x1_input_acceleration
   + 0.2323290868756559 x2_input_acceleration
   - 0.0994519819632967 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1655081785082596 x1_input_acceleration
   + 0.1485898236850618 x2_input_acceleration
   - 0.2115070484436207 x3_input_acceleration >= -0.0135517810993029
 CBF_Constraint_h_left: 0.2144918214917405 x1_input_acceleration
   + 0.1485898236850618 x2_input_acceleration
   - 0.2115070484436207 x3_input_acceleration >= -0.0182052271827336
 CBF_Constraint_h_left: - 0.4855081785082596 x1_input_acceleration
   + 0.1485898236850618 x2_input_acceleration
   - 0.2115070484436207 x3_input_acceleration >= -0.0656330896606245
 Relaxed_CLF_constraint: - 0.0128606624352822 x1_input_acceleration
   + 0.2308982972909867 x2_input_acceleration
   - 0.0998705286163416 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.166701643675463 x1_input_acceleration
   + 0.1457282444010217 x2_input_acceleration
   - 0.2123441420203186 x3_input_acceleration >= -0.0135290484676625
 CBF_Constraint_h_left: 0.213298356324537 x1_input_acceleration
   + 0.1457282444010217 x2_input_acceleration
   - 0.2123441420203186 x3_input_acceleration >= -0.0179557361693246
 CBF_Constraint_h_left: - 0.486701643675463 x1_input_acceleration
   + 0.1457282444010217 x2_input_acceleration
   - 0.2123441420203186 x3_input_acceleration >= -0.0658013114557366
 Relaxed_CLF_constraint: - 0.0134573950188839 x1_input_acceleration
   + 0.2294675076489666 x2_input_acceleration
   - 0.1002890754046905 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1676328483461252 x1_input_acceleration
   + 0.1436276970510747 x2_input_acceleration
   - 0.2119570759877956 x3_input_acceleration >= -0.0134138723165316
 CBF_Constraint_h_left: 0.2123671516538748 x1_input_acceleration
   + 0.1436276970510747 x2_input_acceleration
   - 0.2119570759877956 x3_input_acceleration >= -0.0176636311307678
 CBF_Constraint_h_left: - 0.4876328483461252 x1_input_acceleration
   + 0.1436276970510747 x2_input_acceleration
   - 0.2119570759877956 x3_input_acceleration >= -0.0658351280519116
 Relaxed_CLF_constraint: - 0.0140541275727209 x1_input_acceleration
   + 0.228036717921747 x2_input_acceleration
   - 0.1007076224856275 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1690962032856583 x1_input_acceleration
   + 0.1400203445481298 x2_input_acceleration
   - 0.2140283431119138 x3_input_acceleration >= -0.0135018386270582
 CBF_Constraint_h_left: 0.2111736865458847 x1_input_acceleration
   + 0.1407661178635595 x2_input_acceleration
   - 0.2127941693371945 x3_input_acceleration >= -0.0174226960829159
 CBF_Constraint_h_left: - 0.4888263134541153 x1_input_acceleration
   + 0.1407661178635595 x2_input_acceleration
   - 0.2127941693371945 x3_input_acceleration >= -0.0660119057918563
 Relaxed_CLF_constraint: - 0.0146508601267159 x1_input_acceleration
   + 0.2266059283279894 x2_input_acceleration
   - 0.101126169160327 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1702896683824003 x1_input_acceleration
   + 0.137158765382262 x2_input_acceleration
   - 0.2148654364101785 x3_input_acceleration >= -0.0134945634606777
 CBF_Constraint_h_left: 0.2097103316175997 x1_input_acceleration
   + 0.137158765382262 x2_input_acceleration
   - 0.2148654364101785 x3_input_acceleration >= -0.0172395264680216
 CBF_Constraint_h_left: - 0.4902896683824003 x1_input_acceleration
   + 0.137158765382262 x2_input_acceleration
   - 0.2148654364101785 x3_input_acceleration >= -0.0663409104018617
 Relaxed_CLF_constraint: - 0.015247592675087 x1_input_acceleration
   + 0.2251751387450555 x2_input_acceleration
   - 0.1015447158094593 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.171483133473508 x1_input_acceleration
   + 0.134297186216193 x2_input_acceleration
   - 0.2157025297127407 x3_input_acceleration >= -0.0134924451539889
 CBF_Constraint_h_left: 0.208516866526492 x1_input_acceleration
   + 0.134297186216193 x2_input_acceleration
   - 0.2157025297127407 x3_input_acceleration >= -0.0170106497940224
 CBF_Constraint_h_left: - 0.491483133473508 x1_input_acceleration
   + 0.134297186216193 x2_input_acceleration
   - 0.2157025297127407 x3_input_acceleration >= -0.0665297465097502
 Relaxed_CLF_constraint: - 0.0158443252206408 x1_input_acceleration
   + 0.223744349162021 x2_input_acceleration
   - 0.1019632624607404 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1726580018916455 x1_input_acceleration
   + 0.1314407330598433 x2_input_acceleration
   - 0.2165365235642623 x3_input_acceleration >= -0.0134938794904552
 CBF_Constraint_h_left: 0.2073419981083545 x1_input_acceleration
   + 0.1314407330598433 x2_input_acceleration
   - 0.2165365235642623 x3_input_acceleration >= -0.0167888591310425
 CBF_Constraint_h_left: - 0.4926580018916455 x1_input_acceleration
   + 0.1314407330598433 x2_input_acceleration
   - 0.2165365235642623 x3_input_acceleration >= -0.0667191597931185
 Relaxed_CLF_constraint: - 0.0164410577542945 x1_input_acceleration
   + 0.2223135595841208 x2_input_acceleration
   - 0.1023818091072654 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1736054190156214 x1_input_acceleration
   + 0.1293236159075385 x2_input_acceleration
   - 0.2161452843414351 x3_input_acceleration >= -0.0133955557715075
 CBF_Constraint_h_left: 0.2063945809843786 x1_input_acceleration
   + 0.1293236159075385 x2_input_acceleration
   - 0.2161452843414351 x3_input_acceleration >= -0.0165105261585394
 CBF_Constraint_h_left: - 0.4938514669892542 x1_input_acceleration
   + 0.1285791538942939 x2_input_acceleration
   - 0.2173736168609663 x3_input_acceleration >= -0.0669182898927073
 Relaxed_CLF_constraint: - 0.0170377903030988 x1_input_acceleration
   + 0.2208827700013461 x2_input_acceleration
   - 0.1028003557556174 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1747998377822484 x1_input_acceleration
   + 0.1264666859113821 x2_input_acceleration
   - 0.2169917951593026 x3_input_acceleration >= -0.0134085612751413
 CBF_Constraint_h_left: 0.2052001622177516 x1_input_acceleration
   + 0.1264666859113821 x2_input_acceleration
   - 0.2169917951593026 x3_input_acceleration >= -0.0162965920965141
 CBF_Constraint_h_left: - 0.4947998377822485 x1_input_acceleration
   + 0.1264666859113821 x2_input_acceleration
   - 0.2169917951593026 x3_input_acceleration >= -0.0669765353203011
 Relaxed_CLF_constraint: - 0.0176345228492542 x1_input_acceleration
   + 0.2194519804221217 x2_input_acceleration
   - 0.1032189023976139 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1759713683676828 x1_input_acceleration
   + 0.1236022457280225 x2_input_acceleration
   - 0.2178042121240276 x3_input_acceleration >= -0.0134205281132934
 CBF_Constraint_h_left: 0.2040286316323172 x1_input_acceleration
   + 0.1236022457280225 x2_input_acceleration
   - 0.2178042121240276 x3_input_acceleration >= -0.0160859681234337
 CBF_Constraint_h_left: - 0.4959713683676828 x1_input_acceleration
   + 0.1236022457280225 x2_input_acceleration
   - 0.2178042121240276 x3_input_acceleration >= -0.0671759470521226
 Relaxed_CLF_constraint: - 0.01823125539661 x1_input_acceleration
   + 0.2180211908419165 x2_input_acceleration
   - 0.1036374490414449 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.177149763226539 x1_input_acceleration
   + 0.1207652824860046 x2_input_acceleration
   - 0.2185940164305022 x3_input_acceleration >= -0.0134374090210905
 CBF_Constraint_h_left: 0.202850236773461 x1_input_acceleration
   + 0.1207652824860046 x2_input_acceleration
   - 0.2185940164305022 x3_input_acceleration >= -0.0158789540080481
 CBF_Constraint_h_left: - 0.497149763226539 x1_input_acceleration
   + 0.1207652824860046 x2_input_acceleration
   - 0.2185940164305022 x3_input_acceleration >= -0.0673813711373367
 Relaxed_CLF_constraint: - 0.0188204528260381 x1_input_acceleration
   + 0.2166027092209075 x2_input_acceleration
   - 0.1040323511946823 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1782805770232259 x1_input_acceleration
   + 0.1180109049870149 x2_input_acceleration
   - 0.2192434006976602 x3_input_acceleration >= -0.0134445516472659
 CBF_Constraint_h_left: 0.2017194229767741 x1_input_acceleration
   + 0.1180109049870149 x2_input_acceleration
   - 0.2192434006976602 x3_input_acceleration >= -0.0156712420128529
 CBF_Constraint_h_left: - 0.4982805770232259 x1_input_acceleration
   + 0.1180109049870149 x2_input_acceleration
   - 0.2192434006976602 x3_input_acceleration >= -0.067569443970982
 Relaxed_CLF_constraint: - 0.0193803760970622 x1_input_acceleration
   + 0.2152298120058365 x2_input_acceleration
   - 0.1043404732370137 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1796467726189116 x1_input_acceleration
   + 0.1145545564428164 x2_input_acceleration
   - 0.2209966834308716 x3_input_acceleration >= -0.0135588108504116
 CBF_Constraint_h_left: 0.2006183488410493 x1_input_acceleration
   + 0.1152919851079776 x2_input_acceleration
   - 0.2198093589069458 x3_input_acceleration >= -0.0154640294962324
 CBF_Constraint_h_left: - 0.4993816511589507 x1_input_acceleration
   + 0.1152919851079776 x2_input_acceleration
   - 0.2198093589069458 x3_input_acceleration >= -0.0677476074018651
 Relaxed_CLF_constraint: - 0.0199309131649246 x1_input_acceleration
   + 0.2138703520663178 x2_input_acceleration
   - 0.1046234523416565 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.180747846489287 x1_input_acceleration
   + 0.1118356369499823 x2_input_acceleration
   - 0.2215626409223824 x3_input_acceleration >= -0.0135667493887559
 CBF_Constraint_h_left: 0.199252153510713 x1_input_acceleration
   + 0.1118356369499823 x2_input_acceleration
   - 0.2215626409223824 x3_input_acceleration >= -0.0153246585557914
 CBF_Constraint_h_left: - 0.5007478464892869 x1_input_acceleration
   + 0.1118356369499823 x2_input_acceleration
   - 0.2215626409223824 x3_input_acceleration >= -0.0680864048270418
 Relaxed_CLF_constraint: - 0.0204814501001123 x1_input_acceleration
   + 0.2125108923199008 x2_input_acceleration
   - 0.1049064310874119 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1818298378405209 x1_input_acceleration
   + 0.1091449537534462 x2_input_acceleration
   - 0.2220765508547142 x3_input_acceleration >= -0.0135731763246196
 CBF_Constraint_h_left: 0.1981701621594791 x1_input_acceleration
   + 0.1091449537534462 x2_input_acceleration
   - 0.2220765508547142 x3_input_acceleration >= -0.0151255071349207
 CBF_Constraint_h_left: - 0.5018298378405208 x1_input_acceleration
   + 0.1091449537534462 x2_input_acceleration
   - 0.2220765508547142 x3_input_acceleration >= -0.0682659503791029
 Relaxed_CLF_constraint: - 0.0210224457757293 x1_input_acceleration
   + 0.2111655507216327 x2_input_acceleration
   - 0.1051633860535778 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1828956398525897 x1_input_acceleration
   + 0.1064790446151014 x2_input_acceleration
   - 0.2225454691119584 x3_input_acceleration >= -0.0135787719603737
 CBF_Constraint_h_left: 0.1971043601474103 x1_input_acceleration
   + 0.1064790446151014 x2_input_acceleration
   - 0.2225454691119584 x3_input_acceleration >= -0.0149286003883816
 CBF_Constraint_h_left: - 0.5028956398525897 x1_input_acceleration
   + 0.1064790446151014 x2_input_acceleration
   - 0.2225454691119584 x3_input_acceleration >= -0.068442074336788
 Relaxed_CLF_constraint: - 0.0215553467817637 x1_input_acceleration
   + 0.2098325961524603 x2_input_acceleration
   - 0.1053978451821999 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1839729650230593 x1_input_acceleration
   + 0.1038100919476321 x2_input_acceleration
   - 0.2230139556599321 x3_input_acceleration >= -0.0135894528671604
 CBF_Constraint_h_left: 0.1960270349769407 x1_input_acceleration
   + 0.1038100919476321 x2_input_acceleration
   - 0.2230139556599321 x3_input_acceleration >= -0.0147345895127792
 CBF_Constraint_h_left: - 0.5039729650230593 x1_input_acceleration
   + 0.1038100919476321 x2_input_acceleration
   - 0.2230139556599321 x3_input_acceleration >= -0.0686251272708499
 Relaxed_CLF_constraint: - 0.0220873336467836 x1_input_acceleration
   + 0.2085010404463197 x2_input_acceleration
   - 0.1056297638750405 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1847557820957403 x1_input_acceleration
   + 0.1019265661886934 x2_input_acceleration
   - 0.2222304152383516 x3_input_acceleration >= -0.0134775203424592
 CBF_Constraint_h_left: 0.1949771890956659 x1_input_acceleration
   + 0.1011693487814669 x2_input_acceleration
   - 0.2234377669229219 x3_input_acceleration >= -0.0145439442720038
 CBF_Constraint_h_left: - 0.5050228109043341 x1_input_acceleration
   + 0.1011693487814669 x2_input_acceleration
   - 0.2234377669229219 x3_input_acceleration >= -0.0688019280885207
 Relaxed_CLF_constraint: - 0.022612256587421 x1_input_acceleration
   + 0.207180668863237 x2_input_acceleration
   - 0.1058416695065354 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.186055276244654 x1_input_acceleration
   + 0.0985310667528015 x2_input_acceleration
   - 0.2238739051076747 x3_input_acceleration >= -0.0136111155805199
 CBF_Constraint_h_left: 0.1941979242863519 x1_input_acceleration
   + 0.0992909067643982 x2_input_acceleration
   - 0.2226453341695644 x3_input_acceleration >= -0.0142856156976753
 CBF_Constraint_h_left: - 0.5058020757136481 x1_input_acceleration
   + 0.0992909067643982 x2_input_acceleration
   - 0.2226453341695644 x3_input_acceleration >= -0.0688163421974521
 Relaxed_CLF_constraint: - 0.0231354033963749 x1_input_acceleration
   + 0.2058628391510894 x2_input_acceleration
   - 0.1060491289721418 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1871007527499425 x1_input_acceleration
   + 0.0958965764228442 x2_input_acceleration
   - 0.2242867792355216 x3_input_acceleration >= -0.0136268510972653
 CBF_Constraint_h_left: 0.1928992472500575 x1_input_acceleration
   + 0.0958965764228442 x2_input_acceleration
   - 0.2242867792355216 x3_input_acceleration >= -0.0141777080747762
 CBF_Constraint_h_left: - 0.5071007527499425 x1_input_acceleration
   + 0.0958965764228442 x2_input_acceleration
   - 0.2242867792355216 x3_input_acceleration >= -0.0691629715372561
 Relaxed_CLF_constraint: - 0.0236581416490192 x1_input_acceleration
   + 0.2045455939861108 x2_input_acceleration
   - 0.1062555660360652 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1881269288087271 x1_input_acceleration
   + 0.0932906129298298 x2_input_acceleration
   - 0.2246505009806597 x3_input_acceleration >= -0.0136406818486721
 CBF_Constraint_h_left: 0.1918730711912729 x1_input_acceleration
   + 0.0932906129298298 x2_input_acceleration
   - 0.2246505009806597 x3_input_acceleration >= -0.013996565375014
 CBF_Constraint_h_left: - 0.5081269288087271 x1_input_acceleration
   + 0.0932906129298298 x2_input_acceleration
   - 0.2246505009806597 x3_input_acceleration >= -0.0693409904580685
 Relaxed_CLF_constraint: - 0.0241712296784115 x1_input_acceleration
   + 0.2032426122396035 x2_input_acceleration
   - 0.1064374269086343 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1891530427828926 x1_input_acceleration
   + 0.090684740805952 x2_input_acceleration
   - 0.2250140649156164 x3_input_acceleration >= -0.0136584813047297
 CBF_Constraint_h_left: 0.1908469572171074 x1_input_acceleration
   + 0.090684740805952 x2_input_acceleration
   - 0.2250140649156164 x3_input_acceleration >= -0.0138194031759801
 CBF_Constraint_h_left: - 0.5091530427828926 x1_input_acceleration
   + 0.090684740805952 x2_input_acceleration
   - 0.2250140649156164 x3_input_acceleration >= -0.0695229681499925
 Relaxed_CLF_constraint: - 0.0246842866654942 x1_input_acceleration
   + 0.2019396761776647 x2_input_acceleration
   - 0.1066192088761126 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1901400902498881 x1_input_acceleration
   + 0.0881599662479319 x2_input_acceleration
   - 0.2252540557467156 x3_input_acceleration >= -0.0136662057998541
 CBF_Constraint_h_left: 0.1898599097501119 x1_input_acceleration
   + 0.0881599662479319 x2_input_acceleration
   - 0.2252540557467156 x3_input_acceleration >= -0.0136395886523754
 CBF_Constraint_h_left: - 0.5101400902498882 x1_input_acceleration
   + 0.0881599662479319 x2_input_acceleration
   - 0.2252540557467156 x3_input_acceleration >= -0.0696886202398362
 Relaxed_CLF_constraint: - 0.02517304202741 x1_input_acceleration
   + 0.200673474201389 x2_input_acceleration
   - 0.1067387870591488 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1910841643346964 x1_input_acceleration
   + 0.0856828102749929 x2_input_acceleration
   - 0.2253921403750814 x3_input_acceleration >= -0.0136640796947425
 CBF_Constraint_h_left: 0.1889225113855185 x1_input_acceleration
   + 0.0856856712979421 x2_input_acceleration
   - 0.2253958358630574 x3_input_acceleration >= -0.0134592580995992
 CBF_Constraint_h_left: - 0.5110774886144815 x1_input_acceleration
   + 0.0856856712979421 x2_input_acceleration
   - 0.2253958358630574 x3_input_acceleration >= -0.0698363791146677
 Relaxed_CLF_constraint: - 0.0256417412097067 x1_input_acceleration
   + 0.1994363267263941 x2_input_acceleration
   - 0.1068096771173197 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.192021562689794 x1_input_acceleration
   + 0.0832085152225891 x2_input_acceleration
   - 0.2255339207164137 x3_input_acceleration >= -0.013665371734274
 CBF_Constraint_h_left: 0.187978437310206 x1_input_acceleration
   + 0.0832085152225891 x2_input_acceleration
   - 0.2255339207164137 x3_input_acceleration >= -0.0132812748232131
 CBF_Constraint_h_left: - 0.512021562689794 x1_input_acceleration
   + 0.0832085152225891 x2_input_acceleration
   - 0.2255339207164137 x3_input_acceleration >= -0.069988821764641
 Relaxed_CLF_constraint: - 0.0261104403872555 x1_input_acceleration
   + 0.1981991792001922 x2_input_acceleration
   - 0.1068805672879859 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1926857332625426 x1_input_acceleration
   + 0.0814755829857569 x2_input_acceleration
   - 0.2244696602149306 x3_input_acceleration >= -0.0135381726956998
 CBF_Constraint_h_left: 0.1873142667374574 x1_input_acceleration
   + 0.0814755829857569 x2_input_acceleration
   - 0.2244696602149306 x3_input_acceleration >= -0.0130278833758167
 CBF_Constraint_h_left: - 0.5129594377913512 x1_input_acceleration
   + 0.0807538899468164 x2_input_acceleration
   - 0.225647805623561 x3_input_acceleration >= -0.0701413769358725
 Relaxed_CLF_constraint: - 0.026579139519455 x1_input_acceleration
   + 0.1969620317959179 x2_input_acceleration
   - 0.106951457228437 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.193623131535326 x1_input_acceleration
   + 0.0790012881584805 x2_input_acceleration
   - 0.2246114401311685 x3_input_acceleration >= -0.0135453549085107
 CBF_Constraint_h_left: 0.186376868464674 x1_input_acceleration
   + 0.0790012881584805 x2_input_acceleration
   - 0.2246114401311685 x3_input_acceleration >= -0.0128569599167988
 CBF_Constraint_h_left: - 0.513623131535326 x1_input_acceleration
   + 0.0790012881584805 x2_input_acceleration
   - 0.2246114401311685 x3_input_acceleration >= -0.0701250559541629
 Relaxed_CLF_constraint: - 0.0270478386558467 x1_input_acceleration
   + 0.1957248843822798 x2_input_acceleration
   - 0.1070223471865559 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1945481320740585 x1_input_acceleration
   + 0.076531999803965 x2_input_acceleration
   - 0.2247267562992111 x3_input_acceleration >= -0.0135520594210661
 CBF_Constraint_h_left: 0.1854394701598282 x1_input_acceleration
   + 0.0765269930138038 x2_input_acceleration
   - 0.2247532207614913 x3_input_acceleration >= -0.0126895469988891
 CBF_Constraint_h_left: - 0.5145605298401719 x1_input_acceleration
   + 0.0765269930138038 x2_input_acceleration
   - 0.2247532207614913 x3_input_acceleration >= -0.0702857324429492
 Relaxed_CLF_constraint: - 0.0275165378082696 x1_input_acceleration
   + 0.1944877368099414 x2_input_acceleration
   - 0.1070932375017173 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1957644800753376 x1_input_acceleration
   + 0.0733219452613905 x2_input_acceleration
   - 0.2260915042798634 x3_input_acceleration >= -0.0137043019059033
 CBF_Constraint_h_left: 0.1845144696622112 x1_input_acceleration
   + 0.0740577049964979 x2_input_acceleration
   - 0.2248685361783619 x3_input_acceleration >= -0.0125239979367687
 CBF_Constraint_h_left: - 0.5154855303377888 x1_input_acceleration
   + 0.0740577049964979 x2_input_acceleration
   - 0.2248685361783619 x3_input_acceleration >= -0.0704439335549947
 Relaxed_CLF_constraint: - 0.0279852369401347 x1_input_acceleration
   + 0.1932505894062079 x2_input_acceleration
   - 0.1071641274412927 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1967018778224396 x1_input_acceleration
   + 0.0708476504306697 x2_input_acceleration
   - 0.2262332846129149 x3_input_acceleration >= -0.0137231293442921
 CBF_Constraint_h_left: 0.1835770719151092 x1_input_acceleration
   + 0.0715834101657771 x2_input_acceleration
   - 0.2250103165114134 x3_input_acceleration >= -0.0123635921201134
 CBF_Constraint_h_left: - 0.5164229280848909 x1_input_acceleration
   + 0.0715834101657771 x2_input_acceleration
   - 0.2250103165114134 x3_input_acceleration >= -0.0706116169498252
 Relaxed_CLF_constraint: - 0.0284539358136857 x1_input_acceleration
   + 0.1920134419908475 x2_input_acceleration
   - 0.1072350176078185 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1976392760849717 x1_input_acceleration
   + 0.0683733556224229 x2_input_acceleration
   - 0.2263750644949685 x3_input_acceleration >= -0.0137454672588908
 CBF_Constraint_h_left: 0.1826396736525772 x1_input_acceleration
   + 0.0691091153575303 x2_input_acceleration
   - 0.225152096393467 x3_input_acceleration >= -0.0122066966819483
 CBF_Constraint_h_left: - 0.5173603263474229 x1_input_acceleration
   + 0.0691091153575303 x2_input_acceleration
   - 0.225152096393467 x3_input_acceleration >= -0.0707828109035464
 Relaxed_CLF_constraint: - 0.0289226349449517 x1_input_acceleration
   + 0.1907762945867241 x2_input_acceleration
   - 0.1073059075488453 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1985637997438148 x1_input_acceleration
   + 0.0658976303042357 x2_input_acceleration
   - 0.2265144601878941 x3_input_acceleration >= -0.0137697202306571
 CBF_Constraint_h_left: 0.1814233256529137 x1_input_acceleration
   + 0.0658990608157103 x2_input_acceleration
   - 0.2265168443736852 x3_input_acceleration >= -0.012141747523092
 CBF_Constraint_h_left: - 0.5185766743470863 x1_input_acceleration
   + 0.0658990608157103 x2_input_acceleration
   - 0.2265168443736852 x3_input_acceleration >= -0.0711435835445723
 Relaxed_CLF_constraint: - 0.029391334076009 x1_input_acceleration
   + 0.1895391471833678 x2_input_acceleration
   - 0.1073767974882036 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1995011979943391 x1_input_acceleration
   + 0.063423335505392 x2_input_acceleration
   - 0.2266562400643448 x3_input_acceleration >= -0.013799074661978
 CBF_Constraint_h_left: 0.1804988020056609 x1_input_acceleration
   + 0.063423335505392 x2_input_acceleration
   - 0.2266562400643448 x3_input_acceleration >= -0.0119938470430535
 CBF_Constraint_h_left: - 0.5195011979943391 x1_input_acceleration
   + 0.063423335505392 x2_input_acceleration
   - 0.2266562400643448 x3_input_acceleration >= -0.0713192663410722
 Relaxed_CLF_constraint: - 0.0298600332012712 x1_input_acceleration
   + 0.188301999783946 x2_input_acceleration
   - 0.107447687426429 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2004395499316877 x1_input_acceleration
   + 0.0609440339090956 x2_input_acceleration
   - 0.2267982583598894 x3_input_acceleration >= -0.0138319096102524
 CBF_Constraint_h_left: 0.1795604500683123 x1_input_acceleration
   + 0.0609440339090956 x2_input_acceleration
   - 0.2267982583598894 x3_input_acceleration >= -0.0118483951232318
 CBF_Constraint_h_left: - 0.5204385962573713 x1_input_acceleration
   + 0.0609490406992568 x2_input_acceleration
   - 0.2267980199413103 x3_input_acceleration >= -0.0715021149714505
 Relaxed_CLF_constraint: - 0.0303287323327873 x1_input_acceleration
   + 0.1870648523808783 x2_input_acceleration
   - 0.1075185773649117 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2013769481863208 x1_input_acceleration
   + 0.0584697391059876 x2_input_acceleration
   - 0.2269400382372774 x3_input_acceleration >= -0.0138682916517738
 CBF_Constraint_h_left: 0.1786230518136792 x1_input_acceleration
   + 0.0584697391059876 x2_input_acceleration
   - 0.2269400382372774 x3_input_acceleration >= -0.0117066714963729
 CBF_Constraint_h_left: - 0.5213769481863209 x1_input_acceleration
   + 0.0584697391059876 x2_input_acceleration
   - 0.2269400382372774 x3_input_acceleration >= -0.0716886033615852
 Relaxed_CLF_constraint: - 0.0307974314601038 x1_input_acceleration
   + 0.1858277049793243 x2_input_acceleration
   - 0.1075894673036057 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2020759278690963 x1_input_acceleration
   + 0.0567418136634611 x2_input_acceleration
   - 0.225852293499384 x3_input_acceleration >= -0.0137658931302268
 CBF_Constraint_h_left: 0.1779240721309037 x1_input_acceleration
   + 0.0567418136634611 x2_input_acceleration
   - 0.225852293499384 x3_input_acceleration >= -0.0114714668350985
 CBF_Constraint_h_left: - 0.5220759278690963 x1_input_acceleration
   + 0.0567418136634611 x2_input_acceleration
   - 0.225852293499384 x3_input_acceleration >= -0.0716980415892822
 Relaxed_CLF_constraint: - 0.0312661305910424 x1_input_acceleration
   + 0.1845905575771223 x2_input_acceleration
   - 0.1076603572408724 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2030133257141608 x1_input_acceleration
   + 0.0542675194240732 x2_input_acceleration
   - 0.2259940726418204 x3_input_acceleration >= -0.0138081737378006
 CBF_Constraint_h_left: 0.1769866742858393 x1_input_acceleration
   + 0.0542675194240732 x2_input_acceleration
   - 0.2259940726418204 x3_input_acceleration >= -0.0113356418521101
 CBF_Constraint_h_left: - 0.5230133257141607 x1_input_acceleration
   + 0.0542675194240732 x2_input_acceleration
   - 0.2259940726418204 x3_input_acceleration >= -0.0718903058520663
 Relaxed_CLF_constraint: - 0.0317348295135746 x1_input_acceleration
   + 0.1833534104574284 x2_input_acceleration
   - 0.1077312468120906 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2042096464722226 x1_input_acceleration
   + 0.0510368417934661 x2_input_acceleration
   - 0.2273769402719905 x3_input_acceleration >= -0.0140016529750034
 CBF_Constraint_h_left: 0.1757903535277774 x1_input_acceleration
   + 0.0510368417934661 x2_input_acceleration
   - 0.2273769402719905 x3_input_acceleration >= -0.0113018201452811
 CBF_Constraint_h_left: - 0.5242096464722226 x1_input_acceleration
   + 0.0510368417934661 x2_input_acceleration
   - 0.2273769402719905 x3_input_acceleration >= -0.072275196410559
 Relaxed_CLF_constraint: - 0.0322035286041533 x1_input_acceleration
   + 0.1821162631132247 x2_input_acceleration
   - 0.1078021366704191 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2051513245719102 x1_input_acceleration
   + 0.0485809266653313 x2_input_acceleration
   - 0.2275124896119242 x3_input_acceleration >= -0.0140522763346719
 CBF_Constraint_h_left: 0.1748486754280899 x1_input_acceleration
   + 0.0485809266653313 x2_input_acceleration
   - 0.2275124896119242 x3_input_acceleration >= -0.0111735246660089
 CBF_Constraint_h_left: - 0.5251513245719102 x1_input_acceleration
   + 0.0485809266653313 x2_input_acceleration
   - 0.2275124896119242 x3_input_acceleration >= -0.0724764882661775
 Relaxed_CLF_constraint: - 0.0326722218867852 x1_input_acceleration
   + 0.1808791264338619 x2_input_acceleration
   - 0.1078730107819142 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2057650740925746 x1_input_acceleration
   + 0.046894738509125 x2_input_acceleration
   - 0.2263662358025181 x3_input_acceleration >= -0.0139450137318908
 CBF_Constraint_h_left: 0.1742349259074254 x1_input_acceleration
   + 0.046894738509125 x2_input_acceleration
   - 0.2263662358025181 x3_input_acceleration >= -0.0109496496543016
 CBF_Constraint_h_left: - 0.5257650740925746 x1_input_acceleration
   + 0.046894738509125 x2_input_acceleration
   - 0.2263662358025181 x3_input_acceleration >= -0.0724674255867027
 Relaxed_CLF_constraint: - 0.0331254856546858 x1_input_acceleration
   + 0.1796696426043244 x2_input_acceleration
   - 0.1079032616962724 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2066297735945747 x1_input_acceleration
   + 0.0445470372032762 x2_input_acceleration
   - 0.2263233963379483 x3_input_acceleration >= -0.0139756453973198
 CBF_Constraint_h_left: 0.1733702264054253 x1_input_acceleration
   + 0.0445470372032762 x2_input_acceleration
   - 0.2263233963379483 x3_input_acceleration >= -0.0108159884143506
 CBF_Constraint_h_left: - 0.5266297735945747 x1_input_acceleration
   + 0.0445470372032762 x2_input_acceleration
   - 0.2263233963379483 x3_input_acceleration >= -0.0726364091724518
 Relaxed_CLF_constraint: - 0.0335578354056859 x1_input_acceleration
   + 0.1784957919514 x2_input_acceleration
   - 0.1078818419639876 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2075092534675314 x1_input_acceleration
   + 0.0422012459162994 x2_input_acceleration
   - 0.226252061976005 x3_input_acceleration >= -0.0140077577449835
 CBF_Constraint_h_left: 0.1724907465324686 x1_input_acceleration
   + 0.0422012459162994 x2_input_acceleration
   - 0.226252061976005 x3_input_acceleration >= -0.0106809995861526
 CBF_Constraint_h_left: - 0.5275092534675314 x1_input_acceleration
   + 0.0422012459162994 x2_input_acceleration
   - 0.226252061976005 x3_input_acceleration >= -0.0728092382997886
 Relaxed_CLF_constraint: - 0.0339901843662121 x1_input_acceleration
   + 0.1773219426335952 x2_input_acceleration
   - 0.1078604202931172 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.208373950873826 x1_input_acceleration
   + 0.0398535480067034 x2_input_acceleration
   - 0.2262092176436623 x3_input_acceleration >= -0.014044654709612
 CBF_Constraint_h_left: 0.171626049126174 x1_input_acceleration
   + 0.0398535480067034 x2_input_acceleration
   - 0.2262092176436623 x3_input_acceleration >= -0.0105536040435851
 CBF_Constraint_h_left: - 0.528373950873826 x1_input_acceleration
   + 0.0398535480067034 x2_input_acceleration
   - 0.2262092176436623 x3_input_acceleration >= -0.0729844868494242
 Relaxed_CLF_constraint: - 0.0344225330693594 x1_input_acceleration
   + 0.1761480936787972 x2_input_acceleration
   - 0.1078389981269459 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2092372534481971 x1_input_acceleration
   + 0.0375167586462498 x2_input_acceleration
   - 0.2261716473075483 x3_input_acceleration >= -0.014085337363919
 CBF_Constraint_h_left: 0.1707684685977013 x1_input_acceleration
   + 0.0375160433905125 x2_input_acceleration
   - 0.2261523354026411 x3_input_acceleration >= -0.0104285505467379
 CBF_Constraint_h_left: - 0.5292315314022986 x1_input_acceleration
   + 0.0375160433905125 x2_input_acceleration
   - 0.2261523354026411 x3_input_acceleration >= -0.0731595865375424
 Relaxed_CLF_constraint: - 0.0348513233335957 x1_input_acceleration
   + 0.1749793413707018 x2_input_acceleration
   - 0.1078105570064353 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.21008577090452 x1_input_acceleration
   + 0.0351811660490721 x2_input_acceleration
   - 0.2261158316710578 x3_input_acceleration >= -0.0141255287283532
 CBF_Constraint_h_left: 0.1699051691894742 x1_input_acceleration
   + 0.0351792587004393 x2_input_acceleration
   - 0.2261147587874519 x3_input_acceleration >= -0.010308307725381
 CBF_Constraint_h_left: - 0.5300948308105258 x1_input_acceleration
   + 0.0351792587004393 x2_input_acceleration
   - 0.2261147587874519 x3_input_acceleration >= -0.073341498509065
 Relaxed_CLF_constraint: - 0.0352801120147601 x1_input_acceleration
   + 0.1738105913977965 x2_input_acceleration
   - 0.107782112746387 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2109224740401296 x1_input_acceleration
   + 0.0328744800441517 x2_input_acceleration
   - 0.2260171482531143 x3_input_acceleration >= -0.0141631931994632
 CBF_Constraint_h_left: 0.1690775259598704 x1_input_acceleration
   + 0.0328744800441517 x2_input_acceleration
   - 0.2260171482531143 x3_input_acceleration >= -0.0101879231318385
 CBF_Constraint_h_left: - 0.5309224740401297 x1_input_acceleration
   + 0.0328744800441517 x2_input_acceleration
   - 0.2260171482531143 x3_input_acceleration >= -0.0735107890458839
 Relaxed_CLF_constraint: - 0.0356984635825649 x1_input_acceleration
   + 0.1726572483953363 x2_input_acceleration
   - 0.1077327710374153 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2117591728667898 x1_input_acceleration
   + 0.0305678001844182 x2_input_acceleration
   - 0.2259184566070272 x3_input_acceleration >= -0.0142038716842607
 CBF_Constraint_h_left: 0.1682408271332102 x1_input_acceleration
   + 0.0305678001844182 x2_input_acceleration
   - 0.2259184566070272 x3_input_acceleration >= -0.0100696288395706
 CBF_Constraint_h_left: - 0.5317591728667899 x1_input_acceleration
   + 0.0305678001844182 x2_input_acceleration
   - 0.2259184566070272 x3_input_acceleration >= -0.073685339342947
 Relaxed_CLF_constraint: - 0.036116812995895 x1_input_acceleration
   + 0.1715039084654695 x2_input_acceleration
   - 0.1076834252143717 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2125711183986811 x1_input_acceleration
   + 0.0282967923300007 x2_input_acceleration
   - 0.2257791143933899 x3_input_acceleration >= -0.0142408493324242
 CBF_Constraint_h_left: 0.1674288816013189 x1_input_acceleration
   + 0.0282967923300007 x2_input_acceleration
   - 0.2257791143933899 x3_input_acceleration >= -0.00995233683667477
 CBF_Constraint_h_left: - 0.5325711183986811 x1_input_acceleration
   + 0.0282967923300007 x2_input_acceleration
   - 0.2257791143933899 x3_input_acceleration >= -0.0738522282762132
 Relaxed_CLF_constraint: - 0.0365249315290525 x1_input_acceleration
   + 0.1703661395617593 x2_input_acceleration
   - 0.1076132772703949 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2133873279202273 x1_input_acceleration
   + 0.0260212965472989 x2_input_acceleration
   - 0.2256387622904014 x3_input_acceleration >= -0.0142810276597203
 CBF_Constraint_h_left: 0.1666126720797727 x1_input_acceleration
   + 0.0260212965472989 x2_input_acceleration
   - 0.2256387622904014 x3_input_acceleration >= -0.00983743535487715
 CBF_Constraint_h_left: - 0.5333873279202272 x1_input_acceleration
   + 0.0260212965472989 x2_input_acceleration
   - 0.2256387622904014 x3_input_acceleration >= -0.0740230001269567
 Relaxed_CLF_constraint: - 0.0369330362898256 x1_input_acceleration
   + 0.1692283916704084 x2_input_acceleration
   - 0.1075431012189006 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2144700893481135 x1_input_acceleration
   + 0.0229939478760689 x2_input_acceleration
   - 0.2267043312236892 x3_input_acceleration >= -0.0144802486648738
 CBF_Constraint_h_left: 0.1655299106518865 x1_input_acceleration
   + 0.0229939478760689 x2_input_acceleration
   - 0.2267043312236892 x3_input_acceleration >= -0.00983093168873227
 CBF_Constraint_h_left: - 0.5344700893481135 x1_input_acceleration
   + 0.0229939478760689 x2_input_acceleration
   - 0.2267043312236892 x3_input_acceleration >= -0.074395462960572
 Relaxed_CLF_constraint: - 0.0373411410180509 x1_input_acceleration
   + 0.1680906438293918 x2_input_acceleration
   - 0.1074729250989967 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2152263707893135 x1_input_acceleration
   + 0.0208254736853551 x2_input_acceleration
   - 0.2264154421087723 x3_input_acceleration >= -0.0145050108656673
 CBF_Constraint_h_left: 0.1647736292106865 x1_input_acceleration
   + 0.0208254736853551 x2_input_acceleration
   - 0.2264154421087723 x3_input_acceleration >= -0.00971200041569776
 CBF_Constraint_h_left: - 0.5352263707893135 x1_input_acceleration
   + 0.0208254736853551 x2_input_acceleration
   - 0.2264154421087723 x3_input_acceleration >= -0.0745412301919575
 Relaxed_CLF_constraint: - 0.0377133212741734 x1_input_acceleration
   + 0.1670084332919573 x2_input_acceleration
   - 0.1073300302623024 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.215712252603194 x1_input_acceleration
   + 0.0194053268713664 x2_input_acceleration
   - 0.2249264068587727 x3_input_acceleration >= -0.0143750577841317
 CBF_Constraint_h_left: 0.1640293016570599 x1_input_acceleration
   + 0.0186611032767009 x2_input_acceleration
   - 0.2261295862182087 x3_input_acceleration >= -0.00959710958520619
 CBF_Constraint_h_left: - 0.5359706983429401 x1_input_acceleration
   + 0.0186611032767009 x2_input_acceleration
   - 0.2261295862182087 x3_input_acceleration >= -0.0746868540052352
 Relaxed_CLF_constraint: - 0.0380854850509867 x1_input_acceleration
   + 0.1659262480876302 x2_input_acceleration
   - 0.1071871023170206 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2164201229398262 x1_input_acceleration
   + 0.0172927138163524 x2_input_acceleration
   - 0.2245747710602319 x3_input_acceleration >= -0.0143926338402948
 CBF_Constraint_h_left: 0.1635798770601739 x1_input_acceleration
   + 0.0172927138163524 x2_input_acceleration
   - 0.2245747710602319 x3_input_acceleration >= -0.00937281048172786
 CBF_Constraint_h_left: - 0.5364201229398262 x1_input_acceleration
   + 0.0172927138163524 x2_input_acceleration
   - 0.2245747710602319 x3_input_acceleration >= -0.074619853510667
 Relaxed_CLF_constraint: - 0.0384394202193028 x1_input_acceleration
   + 0.1648699415601232 x2_input_acceleration
   - 0.1070112844177502 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2173797581750921 x1_input_acceleration
   + 0.0144448250980348 x2_input_acceleration
   - 0.2254389416328603 x3_input_acceleration >= -0.0145713321602296
 CBF_Constraint_h_left: 0.1628720118444392 x1_input_acceleration
   + 0.015180107995984 x2_input_acceleration
   - 0.2242231260887319 x3_input_acceleration >= -0.00925843454850706
 CBF_Constraint_h_left: - 0.5371279881555608 x1_input_acceleration
   + 0.015180107995984 x2_input_acceleration
   - 0.2242231260887319 x3_input_acceleration >= -0.0747532304029533
 Relaxed_CLF_constraint: - 0.0387933528271701 x1_input_acceleration
   + 0.163813638649939 x2_input_acceleration
   - 0.1068354619320002 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2180928667128424 x1_input_acceleration
   + 0.0123286457249167 x2_input_acceleration
   - 0.225077518002548 x3_input_acceleration >= -0.0145940957816559
 CBF_Constraint_h_left: 0.1619071332871576 x1_input_acceleration
   + 0.0123286457249167 x2_input_acceleration
   - 0.225077518002548 x3_input_acceleration >= -0.00925645110621581
 CBF_Constraint_h_left: - 0.5380928667128424 x1_input_acceleration
   + 0.0123286457249167 x2_input_acceleration
   - 0.225077518002548 x3_input_acceleration >= -0.0750889544557106
 Relaxed_CLF_constraint: - 0.0391472844916752 x1_input_acceleration
   + 0.1627573371027232 x2_input_acceleration
   - 0.1066596376977156 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2188007300348875 x1_input_acceleration
   + 0.0102160425926797 x2_input_acceleration
   - 0.2247258696137586 x3_input_acceleration >= -0.0146199608659288
 CBF_Constraint_h_left: 0.1611992699651125 x1_input_acceleration
   + 0.0102160425926797 x2_input_acceleration
   - 0.2247258696137586 x3_input_acceleration >= -0.00914782215930017
 CBF_Constraint_h_left: - 0.5388007300348875 x1_input_acceleration
   + 0.0102160425926797 x2_input_acceleration
   - 0.2247258696137586 x3_input_acceleration >= -0.0752280776715108
 Relaxed_CLF_constraint: - 0.0395012161526977 x1_input_acceleration
   + 0.1617010355366047 x2_input_acceleration
   - 0.1064838135033209 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.219498579768671 x1_input_acceleration
   + 0.00811071127538121 x2_input_acceleration
   - 0.2243718369522581 x3_input_acceleration >= -0.0146470328437967
 CBF_Constraint_h_left: 0.160501420231329 x1_input_acceleration
   + 0.00811071127538121 x2_input_acceleration
   - 0.2243718369522581 x3_input_acceleration >= -0.00904230268774924
 CBF_Constraint_h_left: - 0.5395085933489933 x1_input_acceleration
   + 0.00810343950871861 x2_input_acceleration
   - 0.2243742211380491 x3_input_acceleration >= -0.0753697447851467
 Relaxed_CLF_constraint: - 0.0398551478097506 x1_input_acceleration
   + 0.1606447339946241 x2_input_acceleration
   - 0.1063079892654662 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2202064433332123 x1_input_acceleration
   + 0.00599810775528919 x2_input_acceleration
   - 0.2240201890738126 x3_input_acceleration >= -0.0146779750236935
 CBF_Constraint_h_left: 0.1597935566667877 x1_input_acceleration
   + 0.00599810775528919 x2_input_acceleration
   - 0.2240201890738126 x3_input_acceleration >= -0.00893875079038317
 CBF_Constraint_h_left: - 0.5402064433332123 x1_input_acceleration
   + 0.00599810775528919 x2_input_acceleration
   - 0.2240201890738126 x3_input_acceleration >= -0.0755110059570075
 Relaxed_CLF_constraint: - 0.0402090795920213 x1_input_acceleration
   + 0.1595884322345781 x2_input_acceleration
   - 0.1061321653262434 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2209376717087527 x1_input_acceleration
   + 0.00389957128316598 x2_input_acceleration
   - 0.2236776006279066 x3_input_acceleration >= -0.0147150826147336
 CBF_Constraint_h_left: 0.1590856933119993 x1_input_acceleration
   + 0.00388550458699899 x2_input_acceleration
   - 0.2236685407219008 x3_input_acceleration >= -0.00883774276777992
 CBF_Constraint_h_left: - 0.5409143066880007 x1_input_acceleration
   + 0.00388550458699899 x2_input_acceleration
   - 0.2236685407219008 x3_input_acceleration >= -0.0756577501085801
 Relaxed_CLF_constraint: - 0.0405630112694155 x1_input_acceleration
   + 0.158532130650433 x2_input_acceleration
   - 0.1059563411502875 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2216455349440852 x1_input_acceleration
   + 0.00178696829962 x2_input_acceleration
   - 0.2233259520580536 x3_input_acceleration >= -0.0147511043197474
 CBF_Constraint_h_left: 0.1583544650559148 x1_input_acceleration
   + 0.00178696829962 x2_input_acceleration
   - 0.2233259520580536 x3_input_acceleration >= -0.00873845268037121
 CBF_Constraint_h_left: - 0.5416221699233332 x1_input_acceleration
   + 0.001772901603453 x2_input_acceleration
   - 0.2233168921520477 x3_input_acceleration >= -0.0758070381132512
 Relaxed_CLF_constraint: - 0.0409169428870817 x1_input_acceleration
   + 0.15747582915866 x2_input_acceleration
   - 0.105780516865361 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2223448150886313 x1_input_acceleration
   - 3.6008619542093889e-04 x2_input_acceleration
   - 0.2229846747491172 x3_input_acceleration >= -0.0147898779079588
 CBF_Constraint_h_left: 0.1576466018425211 x1_input_acceleration
   - 3.2563471074076311e-04 x2_input_acceleration
   - 0.2229743035409263 x3_input_acceleration >= -0.00864252428750508
 CBF_Constraint_h_left: - 0.542353398157479 x1_input_acceleration
   - 3.2563471074076311e-04 x2_input_acceleration
   - 0.2229743035409263 x3_input_acceleration >= -0.0759662136426227
 Relaxed_CLF_constraint: - 0.0412708744937786 x1_input_acceleration
   + 0.1564195276534797 x2_input_acceleration
   - 0.1056046926067973 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2230359891064833 x1_input_acceleration
   - 0.00246291411361166 x2_input_acceleration
   - 0.2226052504988578 x3_input_acceleration >= -0.0148260539830744
 CBF_Constraint_h_left: 0.1569640108935167 x1_input_acceleration
   - 0.00246291411361166 x2_input_acceleration
   - 0.2226052504988578 x3_input_acceleration >= -0.00854921605284257
 CBF_Constraint_h_left: - 0.5430359891064833 x1_input_acceleration
   - 0.00246291411361166 x2_input_acceleration
   - 0.2226052504988578 x3_input_acceleration >= -0.0761118122401117
 Relaxed_CLF_constraint: - 0.0416248061529732 x1_input_acceleration
   + 0.1553632261135127 x2_input_acceleration
   - 0.1054288683639003 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2237438522975866 x1_input_acceleration
   - 0.00457551700145398 x2_input_acceleration
   - 0.2222536017922352 x3_input_acceleration >= -0.0148697275766041
 CBF_Constraint_h_left: 0.1562561477024134 x1_input_acceleration
   - 0.00457551700145398 x2_input_acceleration
   - 0.2222536017922352 x3_input_acceleration >= -0.00845839564006261
 CBF_Constraint_h_left: - 0.5437438522975866 x1_input_acceleration
   - 0.00457551700145398 x2_input_acceleration
   - 0.2222536017922352 x3_input_acceleration >= -0.0762687439442179
 Relaxed_CLF_constraint: - 0.0419787377485248 x1_input_acceleration
   + 0.1543069246695916 x2_input_acceleration
   - 0.105253044010589 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2244517142966301 x1_input_acceleration
   - 0.00668811794674096 x2_input_acceleration
   - 0.2219019507784192 x3_input_acceleration >= -0.0149159446829084
 CBF_Constraint_h_left: 0.1555482857033699 x1_input_acceleration
   - 0.00668811794674096 x2_input_acceleration
   - 0.2219019507784192 x3_input_acceleration >= -0.00837011896654867
 CBF_Constraint_h_left: - 0.5444517142966301 x1_input_acceleration
   - 0.00668811794674096 x2_input_acceleration
   - 0.2219019507784192 x3_input_acceleration >= -0.0764282189703692
 Relaxed_CLF_constraint: - 0.0423326687480466 x1_input_acceleration
   + 0.1532506241969481 x2_input_acceleration
   - 0.105077218503681 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2249037898088185 x1_input_acceleration
   - 0.008034035225021 x2_input_acceleration
   - 0.2203245324358485 x3_input_acceleration >= -0.0147972899963603
 CBF_Constraint_h_left: 0.1548530232404979 x1_input_acceleration
   - 0.00878028537760889 x2_input_acceleration
   - 0.2215260428652309 x3_input_acceleration >= -0.00828258497139064
 CBF_Constraint_h_left: - 0.5451469767595021 x1_input_acceleration
   - 0.00878028537760889 x2_input_acceleration
   - 0.2215260428652309 x3_input_acceleration >= -0.0765840268372164
 Relaxed_CLF_constraint: - 0.0426803 x1_input_acceleration
   + 0.1522045404815141 x2_input_acceleration
   - 0.1048892645470869 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2255990419742371 x1_input_acceleration
   - 0.01012618545591 x2_input_acceleration
   - 0.2199486045645782 x3_input_acceleration >= -0.0148437140053716
 CBF_Constraint_h_left: 0.154400958025763 x1_input_acceleration
   - 0.01012618545591 x2_input_acceleration
   - 0.2199486045645782 x3_input_acceleration >= -0.00807989603026652
 CBF_Constraint_h_left: - 0.545599041974237 x1_input_acceleration
   - 0.01012618545591 x2_input_acceleration
   - 0.2199486045645782 x3_input_acceleration >= -0.0765395607212495
 Relaxed_CLF_constraint: - 0.0430279260621918 x1_input_acceleration
   + 0.1511584653660696 x2_input_acceleration
   - 0.1047013006114517 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2265514159975386 x1_input_acceleration
   - 0.0129365755332407 x2_input_acceleration
   - 0.2207784343432041 x3_input_acceleration >= -0.0150590040370134
 CBF_Constraint_h_left: 0.1534485840024614 x1_input_acceleration
   - 0.0129365755332407 x2_input_acceleration
   - 0.2207784343432041 x3_input_acceleration >= -0.00811423499748105
 CBF_Constraint_h_left: - 0.5465514159975385 x1_input_acceleration
   - 0.0129365755332407 x2_input_acceleration
   - 0.2207784343432041 x3_input_acceleration >= -0.0769072305966195
 Relaxed_CLF_constraint: - 0.0433744133668113 x1_input_acceleration
   + 0.1501142956608088 x2_input_acceleration
   - 0.1045111091470049 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.227244390449612 x1_input_acceleration
   - 0.0150249146880271 x2_input_acceleration
   - 0.2203980511193845 x3_input_acceleration >= -0.0151102654973553
 CBF_Constraint_h_left: 0.152755609550388 x1_input_acceleration
   - 0.0150249146880271 x2_input_acceleration
   - 0.2203980511193845 x3_input_acceleration >= -0.00803383131192897
 CBF_Constraint_h_left: - 0.547244390449612 x1_input_acceleration
   - 0.0150249146880271 x2_input_acceleration
   - 0.2203980511193845 x3_input_acceleration >= -0.0770693679692932
 Relaxed_CLF_constraint: - 0.0437209005928481 x1_input_acceleration
   + 0.1490701260834156 x2_input_acceleration
   - 0.1043209175350951 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2279192450340521 x1_input_acceleration
   - 0.0171256515121725 x2_input_acceleration
   - 0.2200139722925251 x3_input_acceleration >= -0.0151616545501362
 CBF_Constraint_h_left: 0.1520807549659479 x1_input_acceleration
   - 0.0171256515121725 x2_input_acceleration
   - 0.2200139722925251 x3_input_acceleration >= -0.00795699799366627
 CBF_Constraint_h_left: - 0.5479483321007025 x1_input_acceleration
   - 0.0171133729553488 x2_input_acceleration
   - 0.2200294695001667 x3_input_acceleration >= -0.0772383024085938
 Relaxed_CLF_constraint: - 0.044067387791074 x1_input_acceleration
   + 0.1480259565543995 x2_input_acceleration
   - 0.1041307258656534 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2286122194092047 x1_input_acceleration
   - 0.0192139905523478 x2_input_acceleration
   - 0.2196335889399534 x3_input_acceleration >= -0.0152179094216981
 CBF_Constraint_h_left: 0.1513877805907953 x1_input_acceleration
   - 0.0192139905523478 x2_input_acceleration
   - 0.2196335889399534 x3_input_acceleration >= -0.00788158773394923
 CBF_Constraint_h_left: - 0.5486122194092047 x1_input_acceleration
   - 0.0192139905523478 x2_input_acceleration
   - 0.2196335889399534 x3_input_acceleration >= -0.0773958645271709
 Relaxed_CLF_constraint: - 0.0444138749786503 x1_input_acceleration
   + 0.1469817870343119 x2_input_acceleration
   - 0.1039405341893675 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.229312346300352 x1_input_acceleration
   - 0.0213004222366382 x2_input_acceleration
   - 0.2192454570080661 x3_input_acceleration >= -0.0152766076429769
 CBF_Constraint_h_left: 0.150687653699648 x1_input_acceleration
   - 0.0213004222366382 x2_input_acceleration
   - 0.2192454570080661 x3_input_acceleration >= -0.00780726184590997
 CBF_Constraint_h_left: - 0.5493123463003521 x1_input_acceleration
   - 0.0213004222366382 x2_input_acceleration
   - 0.2192454570080661 x3_input_acceleration >= -0.0775665830510332
 Relaxed_CLF_constraint: - 0.0447603621455374 x1_input_acceleration
   + 0.1459376175178503 x2_input_acceleration
   - 0.1037503425253343 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2300053206601973 x1_input_acceleration
   - 0.0233887612545772 x2_input_acceleration
   - 0.2188650736389353 x3_input_acceleration >= -0.0153378505360001
 CBF_Constraint_h_left: 0.1499946793398027 x1_input_acceleration
   - 0.0233887612545772 x2_input_acceleration
   - 0.2188650736389353 x3_input_acceleration >= -0.0077368396105626
 CBF_Constraint_h_left: - 0.5500053206601974 x1_input_acceleration
   - 0.0233887612545772 x2_input_acceleration
   - 0.2188650736389353 x3_input_acceleration >= -0.0777387018416317
 Relaxed_CLF_constraint: - 0.0451068493254601 x1_input_acceleration
   + 0.1448934480088808 x2_input_acceleration
   - 0.1035601508407689 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2306935264718769 x1_input_acceleration
   - 0.0254900938499985 x2_input_acceleration
   - 0.2185116313528085 x3_input_acceleration >= -0.0154041452692445
 CBF_Constraint_h_left: 0.1492950294363263 x1_input_acceleration
   - 0.0254871136177597 x2_input_acceleration
   - 0.2184931539129281 x3_input_acceleration >= -0.00766946427044415
 CBF_Constraint_h_left: - 0.5507049705636737 x1_input_acceleration
   - 0.0254871136177597 x2_input_acceleration
   - 0.2184931539129281 x3_input_acceleration >= -0.0779162039677299
 Relaxed_CLF_constraint: - 0.0454533364170908 x1_input_acceleration
   + 0.1438492786174507 x2_input_acceleration
   - 0.1033699590479863 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2313865008628093 x1_input_acceleration
   - 0.0275784328986957 x2_input_acceleration
   - 0.2181312480018073 x3_input_acceleration >= -0.0154703810243772
 CBF_Constraint_h_left: 0.1486134991371907 x1_input_acceleration
   - 0.0275784328986957 x2_input_acceleration
   - 0.2181312480018073 x3_input_acceleration >= -0.00760694586044339
 CBF_Constraint_h_left: - 0.5513865008628094 x1_input_acceleration
   - 0.0275784328986957 x2_input_acceleration
   - 0.2181312480018073 x3_input_acceleration >= -0.0780922211624267
 Relaxed_CLF_constraint: - 0.0457998236125571 x1_input_acceleration
   + 0.1428051090931021 x2_input_acceleration
   - 0.1031797673724856 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2321076085103567 x1_input_acceleration
   - 0.0296612881410899 x2_input_acceleration
   - 0.2177477650477597 x3_input_acceleration >= -0.01554195578147
 CBF_Constraint_h_left: 0.1479205248819773 x1_input_acceleration
   - 0.0296667717684093 x2_input_acceleration
   - 0.217750864489288 x3_input_acceleration >= -0.00754400950358771
 CBF_Constraint_h_left: - 0.5520794751180227 x1_input_acceleration
   - 0.0296667717684093 x2_input_acceleration
   - 0.217750864489288 x3_input_acceleration >= -0.0782718257948956
 Relaxed_CLF_constraint: - 0.0461463107401638 x1_input_acceleration
   + 0.1417609396582453 x2_input_acceleration
   - 0.102989575616226 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.23280058289891 x1_input_acceleration
   - 0.0317496271886814 x2_input_acceleration
   - 0.2173673816967549 x3_input_acceleration >= -0.0156131822125988
 CBF_Constraint_h_left: 0.14719941710109 x1_input_acceleration
   - 0.0317496271886814 x2_input_acceleration
   - 0.2173673816967549 x3_input_acceleration >= -0.00748107146180592
 CBF_Constraint_h_left: - 0.55280058289891 x1_input_acceleration
   - 0.0317496271886814 x2_input_acceleration
   - 0.2173673816967549 x3_input_acceleration >= -0.0784612754764244
 Relaxed_CLF_constraint: - 0.0464927979344404 x1_input_acceleration
   + 0.1407167701344496 x2_input_acceleration
   - 0.1027993839407236 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2332045938764142 x1_input_acceleration
   - 0.0330980340425007 x2_input_acceleration
   - 0.2157720171383643 x3_input_acceleration >= -0.0155093564606251
 CBF_Constraint_h_left: 0.1465064428057147 x1_input_acceleration
   - 0.0338379661027424 x2_input_acceleration
   - 0.2169869982174659 x3_input_acceleration >= -0.00742312578224526
 CBF_Constraint_h_left: - 0.5534935571942853 x1_input_acceleration
   - 0.0338379661027424 x2_input_acceleration
   - 0.2169869982174659 x3_input_acceleration >= -0.0786458708002451
 Relaxed_CLF_constraint: - 0.0468392850821281 x1_input_acceleration
   + 0.139672600677419 x2_input_acceleration
   - 0.1026091922010791 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2341803324412715 x1_input_acceleration
   - 0.0359263046462897 x2_input_acceleration
   - 0.2166070911538651 x3_input_acceleration >= -0.0157624398514953
 CBF_Constraint_h_left: 0.1458196675587285 x1_input_acceleration
   - 0.0359263046462897 x2_input_acceleration
   - 0.2166070911538651 x3_input_acceleration >= -0.00736817668765373
 CBF_Constraint_h_left: - 0.5538975680064571 x1_input_acceleration
   - 0.035186372586048 x2_input_acceleration
   - 0.2153916332376054 x3_input_acceleration >= -0.0786085380819987
 Relaxed_CLF_constraint: - 0.0471857721471495 x1_input_acceleration
   + 0.1386284314056454 x2_input_acceleration
   - 0.1024190002506996 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2348587406287528 x1_input_acceleration
   - 0.0379868083868444 x2_input_acceleration
   - 0.2161964617423515 x3_input_acceleration >= -0.0158356339327637
 CBF_Constraint_h_left: 0.1451412593712472 x1_input_acceleration
   - 0.0379868083868444 x2_input_acceleration
   - 0.2161964617423515 x3_input_acceleration >= -0.00731247321330063
 CBF_Constraint_h_left: - 0.5548587406287528 x1_input_acceleration
   - 0.0379868083868444 x2_input_acceleration
   - 0.2161964617423515 x3_input_acceleration >= -0.0790130324333641
 Relaxed_CLF_constraint: - 0.0475249762408901 x1_input_acceleration
   + 0.137598179535368 x2_input_acceleration
   - 0.1022136855449428 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2352309599413932 x1_input_acceleration
   - 0.0392702107594284 x2_input_acceleration
   - 0.2145369912113375 x3_input_acceleration >= -0.0157254686415132
 CBF_Constraint_h_left: 0.1447690400586068 x1_input_acceleration
   - 0.0392702107594284 x2_input_acceleration
   - 0.2145369912113375 x3_input_acceleration >= -0.00713158625264848
 CBF_Constraint_h_left: - 0.5555175390734732 x1_input_acceleration
   - 0.0400132422611984 x2_input_acceleration
   - 0.2157500649418063 x3_input_acceleration >= -0.0791872215742237
 Relaxed_CLF_constraint: - 0.0478543754632504 x1_input_acceleration
   + 0.136584962598191 x2_input_acceleration
   - 0.1019904871446702 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.235883109618238 x1_input_acceleration
   - 0.0412940806524371 x2_input_acceleration
   - 0.2140819896256745 x3_input_acceleration >= -0.0157942851955468
 CBF_Constraint_h_left: 0.1441111683358635 x1_input_acceleration
   - 0.0412950343267535 x2_input_acceleration
   - 0.2140889037644684 x3_input_acceleration >= -0.00707684185356179
 CBF_Constraint_h_left: - 0.5558888316641365 x1_input_acceleration
   - 0.0412950343267535 x2_input_acceleration
   - 0.2140889037644684 x3_input_acceleration >= -0.0791379329360096
 Relaxed_CLF_constraint: - 0.048183311324622 x1_input_acceleration
   + 0.1355725508145285 x2_input_acceleration
   - 0.1017664434212357 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2367667485860793 x1_input_acceleration
   - 0.0439793706092189 x2_input_acceleration
   - 0.2147376939230619 x3_input_acceleration >= -0.0160262388666503
 CBF_Constraint_h_left: 0.1432332514139207 x1_input_acceleration
   - 0.0439793706092189 x2_input_acceleration
   - 0.2147376939230619 x3_input_acceleration >= -0.00714055663529526
 CBF_Constraint_h_left: - 0.5567667485860793 x1_input_acceleration
   - 0.0439793706092189 x2_input_acceleration
   - 0.2147376939230619 x3_input_acceleration >= -0.079508918640423
 Relaxed_CLF_constraint: - 0.0484849406840309 x1_input_acceleration
   + 0.1346027924938524 x2_input_acceleration
   - 0.1014995604243605 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2373699881146705 x1_input_acceleration
   - 0.0459188603247788 x2_input_acceleration
   - 0.2142039022826404 x3_input_acceleration >= -0.0160840911860491
 CBF_Constraint_h_left: 0.1426300118853295 x1_input_acceleration
   - 0.0459188603247788 x2_input_acceleration
   - 0.2142039022826404 x3_input_acceleration >= -0.00708379344426168
 CBF_Constraint_h_left: - 0.5573699881146705 x1_input_acceleration
   - 0.0459188603247788 x2_input_acceleration
   - 0.2142039022826404 x3_input_acceleration >= -0.0796632892843964
 Relaxed_CLF_constraint: - 0.0487865604483265 x1_input_acceleration
   + 0.1336330476360725 x2_input_acceleration
   - 0.1012326646041497 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2376745199735861 x1_input_acceleration
   - 0.0470846467354705 x2_input_acceleration
   - 0.2124580674248662 x3_input_acceleration >= -0.0159611429541969
 CBF_Constraint_h_left: 0.1423254800264139 x1_input_acceleration
   - 0.0470846467354705 x2_input_acceleration
   - 0.2124580674248662 x3_input_acceleration >= -0.00690298415921555
 CBF_Constraint_h_left: - 0.5576745199735861 x1_input_acceleration
   - 0.0470846467354705 x2_input_acceleration
   - 0.2124580674248662 x3_input_acceleration >= -0.0795890661499707
 Relaxed_CLF_constraint: - 0.0490768707350841 x1_input_acceleration
   + 0.1326792347262893 x2_input_acceleration
   - 0.1009505484142763 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2382551379787597 x1_input_acceleration
   - 0.0489922689109816 x2_input_acceleration
   - 0.211893831546101 x3_input_acceleration >= -0.0160161872584028
 CBF_Constraint_h_left: 0.1417448620212403 x1_input_acceleration
   - 0.0489922689109816 x2_input_acceleration
   - 0.211893831546101 x3_input_acceleration >= -0.00684771104243845
 CBF_Constraint_h_left: - 0.5582551379787597 x1_input_acceleration
   - 0.0489922689109816 x2_input_acceleration
   - 0.211893831546101 x3_input_acceleration >= -0.0797370093350043
 Relaxed_CLF_constraint: - 0.0493671797376709 x1_input_acceleration
   + 0.1317254236385338 x2_input_acceleration
   - 0.1006684304748937 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2391017921301996 x1_input_acceleration
   - 0.0516383363297797 x2_input_acceleration
   - 0.2125054232488896 x3_input_acceleration >= -0.0162486849222426
 CBF_Constraint_h_left: 0.1408982078698005 x1_input_acceleration
   - 0.0516383363297797 x2_input_acceleration
   - 0.2125054232488896 x3_input_acceleration >= -0.00691934441750466
 CBF_Constraint_h_left: - 0.5591017921301995 x1_input_acceleration
   - 0.0516383363297797 x2_input_acceleration
   - 0.2125054232488896 x3_input_acceleration >= -0.0801049716630745
 Relaxed_CLF_constraint: - 0.0496574692462522 x1_input_acceleration
   + 0.1307716407035183 x2_input_acceleration
   - 0.1003862861102236 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2396579504283627 x1_input_acceleration
   - 0.053510807710886 x2_input_acceleration
   - 0.2119082759969096 x3_input_acceleration >= -0.0163011142953443
 CBF_Constraint_h_left: 0.1403420495716373 x1_input_acceleration
   - 0.053510807710886 x2_input_acceleration
   - 0.2119082759969096 x3_input_acceleration >= -0.00686610371395543
 CBF_Constraint_h_left: - 0.5596579504283627 x1_input_acceleration
   - 0.053510807710886 x2_input_acceleration
   - 0.2119082759969096 x3_input_acceleration >= -0.0802463863638824
 Relaxed_CLF_constraint: - 0.0499355483953338 x1_input_acceleration
   + 0.1298354050129651 x2_input_acceleration
   - 0.1000877124842336 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2399618611812417 x1_input_acceleration
   - 0.0546402466533534 x2_input_acceleration
   - 0.2101390622018595 x3_input_acceleration >= -0.0161814192097454
 CBF_Constraint_h_left: 0.1400381388187583 x1_input_acceleration
   - 0.0546402466533534 x2_input_acceleration
   - 0.2101390622018595 x3_input_acceleration >= -0.00668866558530952
 CBF_Constraint_h_left: - 0.5599618611812417 x1_input_acceleration
   - 0.0546402466533534 x2_input_acceleration
   - 0.2101390622018595 x3_input_acceleration >= -0.0801753169987441
 Relaxed_CLF_constraint: - 0.050213627200118 x1_input_acceleration
   + 0.1288991697908464 x2_input_acceleration
   - 0.0997891384541402 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2407798029361177 x1_input_acceleration
   - 0.0572522922632952 x2_input_acceleration
   - 0.2107237556592217 x3_input_acceleration >= -0.0164143099176212
 CBF_Constraint_h_left: 0.1392201970638823 x1_input_acceleration
   - 0.0572522922632952 x2_input_acceleration
   - 0.2107237556592217 x3_input_acceleration >= -0.00676614735975885
 CBF_Constraint_h_left: - 0.5605180193362642 x1_input_acceleration
   - 0.0565127178309222 x2_input_acceleration
   - 0.2095419147626153 x3_input_acceleration >= -0.0803204878299173
 Relaxed_CLF_constraint: - 0.0504917062776292 x1_input_acceleration
   + 0.1279629342020621 x2_input_acceleration
   - 0.0994905647345181 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2413116233630112 x1_input_acceleration
   - 0.0591131971082997 x2_input_acceleration
   - 0.210134360432636 x3_input_acceleration >= -0.0164705297692224
 CBF_Constraint_h_left: 0.1386883766369888 x1_input_acceleration
   - 0.0591131971082997 x2_input_acceleration
   - 0.210134360432636 x3_input_acceleration >= -0.00672132133025023
 CBF_Constraint_h_left: - 0.5613116233630112 x1_input_acceleration
   - 0.0591131971082997 x2_input_acceleration
   - 0.210134360432636 x3_input_acceleration >= -0.0806803895073042
 Relaxed_CLF_constraint: - 0.0507588085839715 x1_input_acceleration
   + 0.1270426741738164 x2_input_acceleration
   - 0.0991773896813448 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2418454191416734 x1_input_acceleration
   - 0.0609531340548274 x2_input_acceleration
   - 0.209507483254735 x3_input_acceleration >= -0.0165244692126626
 CBF_Constraint_h_left: 0.1381545808583266 x1_input_acceleration
   - 0.0609531340548274 x2_input_acceleration
   - 0.209507483254735 x3_input_acceleration >= -0.00667383957574468
 CBF_Constraint_h_left: - 0.5618454191416734 x1_input_acceleration
   - 0.0609531340548274 x2_input_acceleration
   - 0.209507483254735 x3_input_acceleration >= -0.0808197362753304
 Relaxed_CLF_constraint: - 0.0510257064733026 x1_input_acceleration
   + 0.1261227057005526 x2_input_acceleration
   - 0.0988639510923943 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2423777868239283 x1_input_acceleration
   - 0.062789496423216 x2_input_acceleration
   - 0.2088730965525976 x3_input_acceleration >= -0.0165793707175544
 CBF_Constraint_h_left: 0.1376222131760717 x1_input_acceleration
   - 0.062789496423216 x2_input_acceleration
   - 0.2088730965525976 x3_input_acceleration >= -0.00662759122100798
 CBF_Constraint_h_left: - 0.5623777868239282 x1_input_acceleration
   - 0.062789496423216 x2_input_acceleration
   - 0.2088730965525976 x3_input_acceleration >= -0.0809598166093829
 Relaxed_CLF_constraint: - 0.0512926055701673 x1_input_acceleration
   + 0.125202736377015 x2_input_acceleration
   - 0.0985505128339465 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2429168193866475 x1_input_acceleration
   - 0.0646194059386372 x2_input_acceleration
   - 0.2082383381891515 x3_input_acceleration >= -0.0166368635641417
 CBF_Constraint_h_left: 0.1370831806133525 x1_input_acceleration
   - 0.0646194059386372 x2_input_acceleration
   - 0.2082383381891515 x3_input_acceleration >= -0.00658266788067871
 CBF_Constraint_h_left: - 0.5626397769977315 x1_input_acceleration
   - 0.0638680297865987 x2_input_acceleration
   - 0.2070086943674352 x3_input_acceleration >= -0.0808738108581473
 Relaxed_CLF_constraint: - 0.0515594992471568 x1_input_acceleration
   + 0.1242827748291432 x2_input_acceleration
   - 0.0982370675587786 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2434127089950239 x1_input_acceleration
   - 0.0664039081603918 x2_input_acceleration
   - 0.2075622223418991 x3_input_acceleration >= -0.0166853255156945
 CBF_Constraint_h_left: 0.1365872910049761 x1_input_acceleration
   - 0.0664039081603918 x2_input_acceleration
   - 0.2075622223418991 x3_input_acceleration >= -0.00653691080663993
 CBF_Constraint_h_left: - 0.5634127089950239 x1_input_acceleration
   - 0.0664039081603918 x2_input_acceleration
   - 0.2075622223418991 x3_input_acceleration >= -0.0812313589548983
 Relaxed_CLF_constraint: - 0.051807444051345 x1_input_acceleration
   + 0.123390523718266 x2_input_acceleration
   - 0.0978990096351524 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2436263108633222 x1_input_acceleration
   - 0.0674547962402855 x2_input_acceleration
   - 0.2056775625814263 x3_input_acceleration >= -0.0165517971575318
 CBF_Constraint_h_left: 0.1363736891366778 x1_input_acceleration
   - 0.0674547962402855 x2_input_acceleration
   - 0.2056775625814263 x3_input_acceleration >= -0.00636279809350055
 CBF_Constraint_h_left: - 0.5636263108633223 x1_input_acceleration
   - 0.0674547962402855 x2_input_acceleration
   - 0.2056775625814263 x3_input_acceleration >= -0.0811320068956634
 Relaxed_CLF_constraint: - 0.0520553887843223 x1_input_acceleration
   + 0.1224982726943714 x2_input_acceleration
   - 0.0975609516436489 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2441303054219497 x1_input_acceleration
   - 0.0692302375565242 x2_input_acceleration
   - 0.2049848762455681 x3_input_acceleration >= -0.0166028078267395
 CBF_Constraint_h_left: 0.1358696945780503 x1_input_acceleration
   - 0.0692302375565242 x2_input_acceleration
   - 0.2049848762455681 x3_input_acceleration >= -0.00631804979656908
 CBF_Constraint_h_left: - 0.5641303054219496 x1_input_acceleration
   - 0.0692302375565242 x2_input_acceleration
   - 0.2049848762455681 x3_input_acceleration >= -0.0812636566942515
 Relaxed_CLF_constraint: - 0.0523033329477913 x1_input_acceleration
   + 0.1216060220832491 x2_input_acceleration
   - 0.0972228935213436 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2448705590356008 x1_input_acceleration
   - 0.0717562531892136 x2_input_acceleration
   - 0.2055321757295605 x3_input_acceleration >= -0.0168385064535723
 CBF_Constraint_h_left: 0.1351294409643992 x1_input_acceleration
   - 0.0717562531892136 x2_input_acceleration
   - 0.2055321757295605 x3_input_acceleration >= -0.00641310023680813
 CBF_Constraint_h_left: - 0.5648705590356009 x1_input_acceleration
   - 0.0717562531892136 x2_input_acceleration
   - 0.2055321757295605 x3_input_acceleration >= -0.0816177958992684
 Relaxed_CLF_constraint: - 0.0525456673962185 x1_input_acceleration
   + 0.1207220998076758 x2_input_acceleration
   - 0.0968776086319922 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2453552279173758 x1_input_acceleration
   - 0.0735240977357922 x2_input_acceleration
   - 0.2048416059559762 x3_input_acceleration >= -0.0168912660862183
 CBF_Constraint_h_left: 0.1346447720826242 x1_input_acceleration
   - 0.0735240977357922 x2_input_acceleration
   - 0.2048416059559762 x3_input_acceleration >= -0.00637377278191689
 CBF_Constraint_h_left: - 0.5653552279173758 x1_input_acceleration
   - 0.0735240977357922 x2_input_acceleration
   - 0.2048416059559762 x3_input_acceleration >= -0.0817481025529984
 Relaxed_CLF_constraint: - 0.0527880018371059 x1_input_acceleration
   + 0.1198381775343866 x2_input_acceleration
   - 0.0965323237452 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2455833544627143 x1_input_acceleration
   - 0.0745397270230681 x2_input_acceleration
   - 0.2029032692874561 x3_input_acceleration >= -0.0167592728953427
 CBF_Constraint_h_left: 0.1341601071461724 x1_input_acceleration
   - 0.0752919376401335 x2_input_acceleration
   - 0.2041510329211841 x3_input_acceleration >= -0.00633636361646615
 CBF_Constraint_h_left: - 0.5658398928538276 x1_input_acceleration
   - 0.0752919376401335 x2_input_acceleration
   - 0.2041510329211841 x3_input_acceleration >= -0.0818803261153058
 Relaxed_CLF_constraint: - 0.0530303343053319 x1_input_acceleration
   + 0.1189542575822159 x2_input_acceleration
   - 0.0961870372278039 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2460731095887043 x1_input_acceleration
   - 0.0762876500601689 x2_input_acceleration
   - 0.2022422168901891 x3_input_acceleration >= -0.0168184237767539
 CBF_Constraint_h_left: 0.1339268904112957 x1_input_acceleration
   - 0.0762876500601689 x2_input_acceleration
   - 0.2022422168901891 x3_input_acceleration >= -0.00616453295490007
 CBF_Constraint_h_left: - 0.5660731095887043 x1_input_acceleration
   - 0.0762876500601689 x2_input_acceleration
   - 0.2022422168901891 x3_input_acceleration >= -0.0817901213109466
 Relaxed_CLF_constraint: - 0.053268536148112 x1_input_acceleration
   + 0.1180766005756894 x2_input_acceleration
   - 0.0958364242638811 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.24652576988352 x1_input_acceleration
   - 0.0780250551320003 x2_input_acceleration
   - 0.2014779262595418 x3_input_acceleration >= -0.0168640548037148
 CBF_Constraint_h_left: 0.13347423011648 x1_input_acceleration
   - 0.0780250551320003 x2_input_acceleration
   - 0.2014779262595418 x3_input_acceleration >= -0.00612415852584603
 CBF_Constraint_h_left: - 0.56652576988352 x1_input_acceleration
   - 0.0780250551320003 x2_input_acceleration
   - 0.2014779262595418 x3_input_acceleration >= -0.081908177985078
 Relaxed_CLF_constraint: - 0.0534986809927855 x1_input_acceleration
   + 0.1172110570859468 x2_input_acceleration
   - 0.0954755578067423 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2472592854750572 x1_input_acceleration
   - 0.0804970244508501 x2_input_acceleration
   - 0.2020144441573944 x3_input_acceleration >= -0.0171066902118244
 CBF_Constraint_h_left: 0.1327407145249428 x1_input_acceleration
   - 0.0804970244508501 x2_input_acceleration
   - 0.2020144441573944 x3_input_acceleration >= -0.00622742597156354
 CBF_Constraint_h_left: - 0.5672592854750572 x1_input_acceleration
   - 0.0804970244508501 x2_input_acceleration
   - 0.2020144441573944 x3_input_acceleration >= -0.0822681758878336
 Relaxed_CLF_constraint: - 0.0537288249427289 x1_input_acceleration
   + 0.116345515293801 x2_input_acceleration
   - 0.0951146897300644 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2474568360535517 x1_input_acceleration
   - 0.0814792351883297 x2_input_acceleration
   - 0.2000381493542163 x3_input_acceleration >= -0.0169722531683923
 CBF_Constraint_h_left: 0.1322804266722784 x1_input_acceleration
   - 0.0822281079452877 x2_input_acceleration
   - 0.2012927079174487 x3_input_acceleration >= -0.0061945818194003
 CBF_Constraint_h_left: - 0.5677195733277216 x1_input_acceleration
   - 0.0822281079452877 x2_input_acceleration
   - 0.2012927079174487 x3_input_acceleration >= -0.0823964324841029
 Relaxed_CLF_constraint: - 0.0539589688690611 x1_input_acceleration
   + 0.1154799735465822 x2_input_acceleration
   - 0.0947538216100915 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2481405068331588 x1_input_acceleration
   - 0.0839236471600838 x2_input_acceleration
   - 0.2005214067461927 x3_input_acceleration >= -0.0172064310618848
 CBF_Constraint_h_left: 0.1321069716519486 x1_input_acceleration
   - 0.0831731054730721 x2_input_acceleration
   - 0.1992854448321302 x3_input_acceleration >= -0.00602117648875588
 CBF_Constraint_h_left: - 0.5678930283480514 x1_input_acceleration
   - 0.0831731054730721 x2_input_acceleration
   - 0.1992854448321302 x3_input_acceleration >= -0.0822837364105739
 Relaxed_CLF_constraint: - 0.0541770650163109 x1_input_acceleration
   + 0.114633038404211 x2_input_acceleration
   - 0.0943774693490485 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2485972026324698 x1_input_acceleration
   - 0.0856208546837898 x2_input_acceleration
   - 0.199786106335282 x3_input_acceleration >= -0.0172614970495211
 CBF_Constraint_h_left: 0.1314027973675302 x1_input_acceleration
   - 0.0856208546837898 x2_input_acceleration
   - 0.199786106335282 x3_input_acceleration >= -0.00612802854935186
 CBF_Constraint_h_left: - 0.5685972026324698 x1_input_acceleration
   - 0.0856208546837898 x2_input_acceleration
   - 0.199786106335282 x3_input_acceleration >= -0.0826370494707163
 Relaxed_CLF_constraint: - 0.054395160917065 x1_input_acceleration
   + 0.1137861035724117 x2_input_acceleration
   - 0.0940011168654559 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2487563482049418 x1_input_acceleration
   - 0.086569779391328 x2_input_acceleration
   - 0.1977915932209805 x3_input_acceleration >= -0.0171238904562538
 CBF_Constraint_h_left: 0.1309666094061422 x1_input_acceleration
   - 0.0873147182417309 x2_input_acceleration
   - 0.199033396390231 x3_input_acceleration >= -0.00609760141990116
 CBF_Constraint_h_left: - 0.5690333905938578 x1_input_acceleration
   - 0.0873147182417309 x2_input_acceleration
   - 0.199033396390231 x3_input_acceleration >= -0.0827592881277514
 Relaxed_CLF_constraint: - 0.0546132548977591 x1_input_acceleration
   + 0.1129391717934412 x2_input_acceleration
   - 0.0936247618929304 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2491758180464468 x1_input_acceleration
   - 0.0882371169365668 x2_input_acceleration
   - 0.1970171828207709 x3_input_acceleration >= -0.0171725368577566
 CBF_Constraint_h_left: 0.1308241819535532 x1_input_acceleration
   - 0.0882371169365668 x2_input_acceleration
   - 0.1970171828207709 x3_input_acceleration >= -0.00592913142893171
 CBF_Constraint_h_left: - 0.5691758180464468 x1_input_acceleration
   - 0.0882371169365668 x2_input_acceleration
   - 0.1970171828207709 x3_input_acceleration >= -0.0826406677451881
 Relaxed_CLF_constraint: - 0.0548229898185116 x1_input_acceleration
   + 0.1121055030208218 x2_input_acceleration
   - 0.0932375566928256 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2495958562952944 x1_input_acceleration
   - 0.0898797270687034 x2_input_acceleration
   - 0.1962401155404851 x3_input_acceleration >= -0.0172216599412672
 CBF_Constraint_h_left: 0.1304041437047056 x1_input_acceleration
   - 0.0898797270687034 x2_input_acceleration
   - 0.1962401155404851 x3_input_acceleration >= -0.00589844724516126
 CBF_Constraint_h_left: - 0.569587750063605 x1_input_acceleration
   - 0.0898933169277122 x2_input_acceleration
   - 0.196234035866718 x3_input_acceleration >= -0.0827547025708315
 Relaxed_CLF_constraint: - 0.0550289558270907 x1_input_acceleration
   + 0.1112774030252491 x2_input_acceleration
   - 0.0928459832157992 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2502728925338538 x1_input_acceleration
   - 0.0922896616954836 x2_input_acceleration
   - 0.1966818237579298 x3_input_acceleration >= -0.0174594105474685
 CBF_Constraint_h_left: 0.1297271074661462 x1_input_acceleration
   - 0.0922896616954836 x2_input_acceleration
   - 0.1966818237579298 x3_input_acceleration >= -0.0060075609660363
 CBF_Constraint_h_left: - 0.5702728925338538 x1_input_acceleration
   - 0.0922896616954836 x2_input_acceleration
   - 0.1966818237579298 x3_input_acceleration >= -0.0831030733528851
 Relaxed_CLF_constraint: - 0.0552349132163899 x1_input_acceleration
   + 0.1104493158807738 x2_input_acceleration
   - 0.0924543995994544 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2503915522559149 x1_input_acceleration
   - 0.093207572751795 x2_input_acceleration
   - 0.194667820632974 x3_input_acceleration >= -0.0173197853623499
 CBF_Constraint_h_left: 0.1296084477440851 x1_input_acceleration
   - 0.093207572751795 x2_input_acceleration
   - 0.194667820632974 x3_input_acceleration >= -0.00584539043372603
 CBF_Constraint_h_left: - 0.5706848071082099 x1_input_acceleration
   - 0.093945835881983 x2_input_acceleration
   - 0.1958986565475858 x3_input_acceleration >= -0.0832208131952121
 Relaxed_CLF_constraint: - 0.0554408705035679 x1_input_acceleration
   + 0.1096212287875241 x2_input_acceleration
   - 0.0920628159942824 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2510904685635135 x1_input_acceleration
   - 0.0955966752850507 x2_input_acceleration
   - 0.1950797025570055 x3_input_acceleration >= -0.0175603095196825
 CBF_Constraint_h_left: 0.1289095314364866 x1_input_acceleration
   - 0.0955966752850507 x2_input_acceleration
   - 0.1950797025570055 x3_input_acceleration >= -0.00595312049261492
 CBF_Constraint_h_left: - 0.5710904685635134 x1_input_acceleration
   - 0.0955966752850507 x2_input_acceleration
   - 0.1950797025570055 x3_input_acceleration >= -0.0833347844898446
 Relaxed_CLF_constraint: - 0.0556465622541689 x1_input_acceleration
   + 0.1087935441094888 x2_input_acceleration
   - 0.091670922369201 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2514525390854707 x1_input_acceleration
   - 0.0971717317633032 x2_input_acceleration
   - 0.1942597146639645 x3_input_acceleration >= -0.0176018904019336
 CBF_Constraint_h_left: 0.1285474609145293 x1_input_acceleration
   - 0.0971717317633032 x2_input_acceleration
   - 0.1942597146639645 x3_input_acceleration >= -0.00592590797569417
 CBF_Constraint_h_left: - 0.5714525390854707 x1_input_acceleration
   - 0.0971717317633032 x2_input_acceleration
   - 0.1942597146639645 x3_input_acceleration >= -0.0834342966556089
 Relaxed_CLF_constraint: - 0.0558225907249864 x1_input_acceleration
   + 0.1080108438465893 x2_input_acceleration
   - 0.0912443583314329 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2518045911596628 x1_input_acceleration
   - 0.0987371252487882 x2_input_acceleration
   - 0.1934065813345246 x3_input_acceleration >= -0.017640169433747
 CBF_Constraint_h_left: 0.1281954088403372 x1_input_acceleration
   - 0.0987371252487882 x2_input_acceleration
   - 0.1934065813345246 x3_input_acceleration >= -0.00589729711341104
 CBF_Constraint_h_left: - 0.5718045911596628 x1_input_acceleration
   - 0.0987371252487882 x2_input_acceleration
   - 0.1934065813345246 x3_input_acceleration >= -0.083528904019293
 Relaxed_CLF_constraint: - 0.0559986167620824 x1_input_acceleration
   + 0.1072281471038468 x2_input_acceleration
   - 0.0908177916667129 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2518705410879704 x1_input_acceleration
   - 0.0995503081047376 x2_input_acceleration
   - 0.1913307181873682 x3_input_acceleration >= -0.0174891192584473
 CBF_Constraint_h_left: 0.1281294589120296 x1_input_acceleration
   - 0.0995503081047376 x2_input_acceleration
   - 0.1913307181873682 x3_input_acceleration >= -0.00573371645173294
 CBF_Constraint_h_left: - 0.5718705410879704 x1_input_acceleration
   - 0.0995503081047376 x2_input_acceleration
   - 0.1913307181873682 x3_input_acceleration >= -0.0833884058325226
 Relaxed_CLF_constraint: - 0.0561746428736972 x1_input_acceleration
   + 0.1064454503673394 x2_input_acceleration
   - 0.0903912249345959 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2525029736609787 x1_input_acceleration
   - 0.101840136493999 x2_input_acceleration
   - 0.1916638363705304 x3_input_acceleration >= -0.0177160478202557
 CBF_Constraint_h_left: 0.1274970263390213 x1_input_acceleration
   - 0.101840136493999 x2_input_acceleration
   - 0.1916638363705304 x3_input_acceleration >= -0.00584048282466972
 CBF_Constraint_h_left: - 0.5725029736609788 x1_input_acceleration
   - 0.101840136493999 x2_input_acceleration
   - 0.1916638363705304 x3_input_acceleration >= -0.0837165236060123
 Relaxed_CLF_constraint: - 0.0563506690356896 x1_input_acceleration
   + 0.1056627535990087 x2_input_acceleration
   - 0.0899646582060171 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2528388142278836 x1_input_acceleration
   - 0.103424604778406 x2_input_acceleration
   - 0.1908264396100713 x3_input_acceleration >= -0.017759711226997
 CBF_Constraint_h_left: 0.1271611857721165 x1_input_acceleration
   - 0.103424604778406 x2_input_acceleration
   - 0.1908264396100713 x3_input_acceleration >= -0.0058203365236991
 CBF_Constraint_h_left: - 0.5728388142278835 x1_input_acceleration
   - 0.103424604778406 x2_input_acceleration
   - 0.1908264396100713 x3_input_acceleration >= -0.0838139215034583
 Relaxed_CLF_constraint: - 0.0565266955508315 x1_input_acceleration
   + 0.1048800561999693 x2_input_acceleration
   - 0.0895380920126772 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2532156618505079 x1_input_acceleration
   - 0.1049883293415631 x2_input_acceleration
   - 0.1899926181725645 x3_input_acceleration >= -0.0178094789160973
 CBF_Constraint_h_left: 0.1267843381494921 x1_input_acceleration
   - 0.1049883293415631 x2_input_acceleration
   - 0.1899926181725645 x3_input_acceleration >= -0.00579850316450078
 CBF_Constraint_h_left: - 0.5732156618505079 x1_input_acceleration
   - 0.1049883293415631 x2_input_acceleration
   - 0.1899926181725645 x3_input_acceleration >= -0.0839239848121785
 Relaxed_CLF_constraint: - 0.0567027215960304 x1_input_acceleration
   + 0.1040973594533639 x2_input_acceleration
   - 0.0891115253414702 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2530908727703715 x1_input_acceleration
   - 0.1057512054598515 x2_input_acceleration
   - 0.1874488613886759 x3_input_acceleration >= -0.0175938457429478
 CBF_Constraint_h_left: 0.1269091272296286 x1_input_acceleration
   - 0.1057512054598515 x2_input_acceleration
   - 0.1874488613886759 x3_input_acceleration >= -0.00560658
 CBF_Constraint_h_left: - 0.5730908727703714 x1_input_acceleration
   - 0.1057512054598515 x2_input_acceleration
   - 0.1874488613886759 x3_input_acceleration >= -0.0836883853862072
 Relaxed_CLF_constraint: - 0.0568787456350637 x1_input_acceleration
   + 0.1033146629255918 x2_input_acceleration
   - 0.0886849600217305 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2536541635244041 x1_input_acceleration
   - 0.1073795414445554 x2_input_acceleration
   - 0.1870686314259963 x3_input_acceleration >= -0.0177163683644258
 CBF_Constraint_h_left: 0.126345836475596 x1_input_acceleration
   - 0.1073795414445554 x2_input_acceleration
   - 0.1870686314259963 x3_input_acceleration >= -0.00562207729478907
 CBF_Constraint_h_left: - 0.573654163524404 x1_input_acceleration
   - 0.1073795414445554 x2_input_acceleration
   - 0.1870686314259963 x3_input_acceleration >= -0.0839010345283305
 Relaxed_CLF_constraint: - 0.057054771581538 x1_input_acceleration
   + 0.1025319661856812 x2_input_acceleration
   - 0.0882583934145667 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2542422482567515 x1_input_acceleration
   - 0.1096876085903606 x2_input_acceleration
   - 0.1874141485900311 x3_input_acceleration >= -0.0179486388421615
 CBF_Constraint_h_left: 0.1257577517432485 x1_input_acceleration
   - 0.1096876085903606 x2_input_acceleration
   - 0.1874141485900311 x3_input_acceleration >= -0.00574261167337874
 CBF_Constraint_h_left: - 0.574006213863441 x1_input_acceleration
   - 0.1089449347164593 x2_input_acceleration
   - 0.186215499183598 x3_input_acceleration >= -0.0840070861226031
 Relaxed_CLF_constraint: - 0.0572307967510565 x1_input_acceleration
   + 0.1017492695497293 x2_input_acceleration
   - 0.0878318272933676 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2543349010866356 x1_input_acceleration
   - 0.1105161694433849 x2_input_acceleration
   - 0.1853534250034211 x3_input_acceleration >= -0.0178139894449216
 CBF_Constraint_h_left: 0.1256650989133644 x1_input_acceleration
   - 0.1105161694433849 x2_input_acceleration
   - 0.1853534250034211 x3_input_acceleration >= -0.00559035823846087
 CBF_Constraint_h_left: - 0.5743349010866355 x1_input_acceleration
   - 0.1105161694433849 x2_input_acceleration
   - 0.1853534250034211 x3_input_acceleration >= -0.0841075736187833
 Relaxed_CLF_constraint: - 0.0574068228730298 x1_input_acceleration
   + 0.1009665728138605 x2_input_acceleration
   - 0.0874052605516373 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2546883838424797 x1_input_acceleration
   - 0.1120827550028114 x2_input_acceleration
   - 0.1845269943948321 x3_input_acceleration >= -0.0178697321234262
 CBF_Constraint_h_left: 0.1253130466689949 x1_input_acceleration
   - 0.1120815629099159 x2_input_acceleration
   - 0.1845002915139727 x3_input_acceleration >= -0.005576498494634
 CBF_Constraint_h_left: - 0.5746869533310052 x1_input_acceleration
   - 0.1120815629099159 x2_input_acceleration
   - 0.1845002915139727 x3_input_acceleration >= -0.0842169321604858
 Relaxed_CLF_constraint: - 0.0575828489952146 x1_input_acceleration
   + 0.100183876080595 x2_input_acceleration
   - 0.0869786938069131 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2550404350127757 x1_input_acceleration
   - 0.113648148338667 x2_input_acceleration
   - 0.1836738615536488 x3_input_acceleration >= -0.0179244031325856
 CBF_Constraint_h_left: 0.1249595649872243 x1_input_acceleration
   - 0.113648148338667 x2_input_acceleration
   - 0.1836738615536488 x3_input_acceleration >= -0.00556672048015823
 CBF_Constraint_h_left: - 0.5750404350127757 x1_input_acceleration
   - 0.113648148338667 x2_input_acceleration
   - 0.1836738615536488 x3_input_acceleration >= -0.0843308727346297
 Relaxed_CLF_constraint: - 0.0577588745803625 x1_input_acceleration
   + 0.0994011794126672 x2_input_acceleration
   - 0.0865521273863214 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.255679065941902 x1_input_acceleration
   - 0.1159462018253636 x2_input_acceleration
   - 0.184022119299919 x3_input_acceleration >= -0.0181698617175713
 CBF_Constraint_h_left: 0.124320934058098 x1_input_acceleration
   - 0.1159462018253636 x2_input_acceleration
   - 0.184022119299919 x3_input_acceleration >= -0.00569083918860988
 CBF_Constraint_h_left: - 0.575679065941902 x1_input_acceleration
   - 0.1159462018253636 x2_input_acceleration
   - 0.184022119299919 x3_input_acceleration >= -0.0846785122682756
 Relaxed_CLF_constraint: - 0.0579349004788857 x1_input_acceleration
   + 0.0986184828161084 x2_input_acceleration
   - 0.0861255606494102 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.256010135614908 x1_input_acceleration
   - 0.1175036068627616 x2_input_acceleration
   - 0.183146693328997 x3_input_acceleration >= -0.0182227496101669
 CBF_Constraint_h_left: 0.123989864385092 x1_input_acceleration
   - 0.1175036068627616 x2_input_acceleration
   - 0.183146693328997 x3_input_acceleration >= -0.0056808238433344
 CBF_Constraint_h_left: - 0.576010135614908 x1_input_acceleration
   - 0.1175036068627616 x2_input_acceleration
   - 0.183146693328997 x3_input_acceleration >= -0.0847843713085522
 Relaxed_CLF_constraint: - 0.0581109257328691 x1_input_acceleration
   + 0.0978357867862095 x2_input_acceleration
   - 0.0856989937325221 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2561113583774096 x1_input_acceleration
   - 0.1183330995975966 x2_input_acceleration
   - 0.181087265563041 x3_input_acceleration >= -0.018097087024849
 CBF_Constraint_h_left: 0.1238886416225904 x1_input_acceleration
   - 0.1183330995975966 x2_input_acceleration
   - 0.181087265563041 x3_input_acceleration >= -0.00553592893314117
 CBF_Constraint_h_left: - 0.5763621747226244 x1_input_acceleration
   - 0.1190689785419936 x2_input_acceleration
   - 0.1822935443640053 x3_input_acceleration >= -0.0849004286047046
 Relaxed_CLF_constraint: - 0.0582869452867273 x1_input_acceleration
   + 0.0970531009465935 x2_input_acceleration
   - 0.0852724192500263 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2564009001386482 x1_input_acceleration
   - 0.1197993890119873 x2_input_acceleration
   - 0.1801657516528296 x3_input_acceleration >= -0.0181382533170459
 CBF_Constraint_h_left: 0.1235990998613517 x1_input_acceleration
   - 0.1197993890119873 x2_input_acceleration
   - 0.1801657516528296 x3_input_acceleration >= -0.00552208229070273
 CBF_Constraint_h_left: - 0.5764009001386483 x1_input_acceleration
   - 0.1197993890119873 x2_input_acceleration
   - 0.1801657516528296 x3_input_acceleration >= -0.0847623973392296
 Relaxed_CLF_constraint: - 0.0584317161673467 x1_input_acceleration
   + 0.0963199562393982 x2_input_acceleration
   - 0.0848116622949207 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2569327239419641 x1_input_acceleration
   - 0.1220070988522328 x2_input_acceleration
   - 0.1804565167910401 x3_input_acceleration >= -0.0183661778137327
 CBF_Constraint_h_left: 0.1230672760580359 x1_input_acceleration
   - 0.1220070988522328 x2_input_acceleration
   - 0.1804565167910401 x3_input_acceleration >= -0.00564896026475957
 CBF_Constraint_h_left: - 0.5769327239419642 x1_input_acceleration
   - 0.1220070988522328 x2_input_acceleration
   - 0.1804565167910401 x3_input_acceleration >= -0.085075413644447
 Relaxed_CLF_constraint: - 0.0585755577565046 x1_input_acceleration
   + 0.0955882131166081 x2_input_acceleration
   - 0.0843499715569885 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2572390024503484 x1_input_acceleration
   - 0.1234934719938434 x2_input_acceleration
   - 0.1795510162042759 x3_input_acceleration >= -0.0184152773566832
 CBF_Constraint_h_left: 0.1227609975496516 x1_input_acceleration
   - 0.1234934719938434 x2_input_acceleration
   - 0.1795510162042759 x3_input_acceleration >= -0.00563986689111697
 CBF_Constraint_h_left: - 0.5772390024503484 x1_input_acceleration
   - 0.1234934719938434 x2_input_acceleration
   - 0.1795510162042759 x3_input_acceleration >= -0.0851735177487389
 Relaxed_CLF_constraint: - 0.0587193986861118 x1_input_acceleration
   + 0.0948564706375997 x2_input_acceleration
   - 0.0838882805668901 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.257540511303924 x1_input_acceleration
   - 0.1249472998500745 x2_input_acceleration
   - 0.1786384819324168 x3_input_acceleration >= -0.0184626624824074
 CBF_Constraint_h_left: 0.122459488696076 x1_input_acceleration
   - 0.1249472998500745 x2_input_acceleration
   - 0.1786384819324168 x3_input_acceleration >= -0.00562996533466179
 CBF_Constraint_h_left: - 0.5775266830263361 x1_input_acceleration
   - 0.1249569558025281 x2_input_acceleration
   - 0.1786276338870677 x3_input_acceleration >= -0.0852647854997323
 Relaxed_CLF_constraint: - 0.0588632389741056 x1_input_acceleration
   + 0.0941247287332574 x2_input_acceleration
   - 0.083426589408286 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2578037459794922 x1_input_acceleration
   - 0.1263629910634503 x2_input_acceleration
   - 0.1776959478966411 x3_input_acceleration >= -0.0185015567126115
 CBF_Constraint_h_left: 0.1221962540205078 x1_input_acceleration
   - 0.1263629910634503 x2_input_acceleration
   - 0.1776959478966411 x3_input_acceleration >= -0.00561884497650797
 CBF_Constraint_h_left: - 0.5778037459794922 x1_input_acceleration
   - 0.1263629910634503 x2_input_acceleration
   - 0.1776959478966411 x3_input_acceleration >= -0.0853501560693302
 Relaxed_CLF_constraint: - 0.0589991478463136 x1_input_acceleration
   + 0.0934052602208383 x2_input_acceleration
   - 0.0829568721111623 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2580766225955431 x1_input_acceleration
   - 0.127787373129971 x2_input_acceleration
   - 0.176735764594703 x3_input_acceleration >= -0.0185421715871638
 CBF_Constraint_h_left: 0.1219233774044569 x1_input_acceleration
   - 0.127787373129971 x2_input_acceleration
   - 0.176735764594703 x3_input_acceleration >= -0.00560761329401059
 CBF_Constraint_h_left: - 0.578064701666588 x1_input_acceleration
   - 0.1277853465720487 x2_input_acceleration
   - 0.1767457781750253 x3_input_acceleration >= -0.0854317410535288
 Relaxed_CLF_constraint: - 0.0591296256898615 x1_input_acceleration
   + 0.0926940824665391 x2_input_acceleration
   - 0.0824817872503544 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2580529066393487 x1_input_acceleration
   - 0.1284736022746416 x2_input_acceleration
   - 0.1745619011901939 x3_input_acceleration >= -0.0183921566133935
 CBF_Constraint_h_left: 0.1216624215772041 x1_input_acceleration
   - 0.1292097196376176 x2_input_acceleration
   - 0.1757855845474327 x3_input_acceleration >= -0.00559936705188592
 CBF_Constraint_h_left: - 0.578337578422796 x1_input_acceleration
   - 0.1292097196376176 x2_input_acceleration
   - 0.1757855845474327 x3_input_acceleration >= -0.0855175194998645
 Relaxed_CLF_constraint: - 0.0592601036034879 x1_input_acceleration
   + 0.0919829092127158 x2_input_acceleration
   - 0.0820066972267192 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2585619904369585 x1_input_acceleration
   - 0.1306489087953899 x2_input_acceleration
   - 0.1748458787922864 x3_input_acceleration >= -0.0186236303996987
 CBF_Constraint_h_left: 0.1214380095630415 x1_input_acceleration
   - 0.1306489087953899 x2_input_acceleration
   - 0.1748458787922864 x3_input_acceleration >= -0.00559685221667659
 CBF_Constraint_h_left: - 0.5785619904369586 x1_input_acceleration
   - 0.1306489087953899 x2_input_acceleration
   - 0.1748458787922864 x3_input_acceleration >= -0.0855935488696121
 Relaxed_CLF_constraint: - 0.0593763627264139 x1_input_acceleration
   + 0.0912701691679788 x2_input_acceleration
   - 0.0815438776972296 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2586306516555782 x1_input_acceleration
   - 0.1320958820887563 x2_input_acceleration
   - 0.1740671008766236 x3_input_acceleration >= -0.0186596229120471
 CBF_Constraint_h_left: 0.1213693483444218 x1_input_acceleration
   - 0.1320958820887563 x2_input_acceleration
   - 0.1740671008766236 x3_input_acceleration >= -0.00561979909748721
 CBF_Constraint_h_left: - 0.5786306516555781 x1_input_acceleration
   - 0.1320958820887563 x2_input_acceleration
   - 0.1740671008766236 x3_input_acceleration >= -0.0856405271769396
 Relaxed_CLF_constraint: - 0.0594164153816222 x1_input_acceleration
   + 0.090550020381403 x2_input_acceleration
   - 0.0811457268566162 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2585747877051124 x1_input_acceleration
   - 0.1335119283803532 x2_input_acceleration
   - 0.1733446486629951 x3_input_acceleration >= -0.0186836807691704
 CBF_Constraint_h_left: 0.1214252122948876 x1_input_acceleration
   - 0.1335119283803532 x2_input_acceleration
   - 0.1733446486629951 x3_input_acceleration >= -0.00565447110519907
 CBF_Constraint_h_left: - 0.5785747877051124 x1_input_acceleration
   - 0.1335119283803532 x2_input_acceleration
   - 0.1733446486629951 x3_input_acceleration >= -0.0856556468019884
 Relaxed_CLF_constraint: - 0.0593884834063893 x1_input_acceleration
   + 0.0898419972356046 x2_input_acceleration
   - 0.080784500749802 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2581558485800962 x1_input_acceleration
   - 0.1341730803800391 x2_input_acceleration
   - 0.1714578984533046 x3_input_acceleration >= -0.0185111671492004
 CBF_Constraint_h_left: 0.1215580491249819 x1_input_acceleration
   - 0.1349360198331641 x2_input_acceleration
   - 0.1726881383214686 x3_input_acceleration >= -0.00570132046810556
 CBF_Constraint_h_left: - 0.5784419508750182 x1_input_acceleration
   - 0.1349360198331641 x2_input_acceleration
   - 0.1726881383214686 x3_input_acceleration >= -0.0856560032743619
 Relaxed_CLF_constraint: - 0.0593220649913421 x1_input_acceleration
   + 0.0891299515091992 x2_input_acceleration
   - 0.0804562455790387 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2579906863851958 x1_input_acceleration
   - 0.1356092672411003 x2_input_acceleration
   - 0.1708427360474253 x3_input_acceleration >= -0.0185340770208357
 CBF_Constraint_h_left: 0.1220093136148042 x1_input_acceleration
   - 0.1356092672411003 x2_input_acceleration
   - 0.1708427360474253 x3_input_acceleration >= -0.00561584660764851
 CBF_Constraint_h_left: - 0.5779906863851958 x1_input_acceleration
   - 0.1356092672411003 x2_input_acceleration
   - 0.1708427360474253 x3_input_acceleration >= -0.0854125868424671
 Relaxed_CLF_constraint: - 0.0592294703135697 x1_input_acceleration
   + 0.0884151363341312 x2_input_acceleration
   - 0.0801504525154424 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2580232624810597 x1_input_acceleration
   - 0.1377732103328372 x2_input_acceleration
   - 0.1714406890541802 x3_input_acceleration >= -0.0187373428325395
 CBF_Constraint_h_left: 0.1219767375189404 x1_input_acceleration
   - 0.1377732103328372 x2_input_acceleration
   - 0.1714406890541802 x3_input_acceleration >= -0.00581292296113819
 CBF_Constraint_h_left: - 0.5780232624810596 x1_input_acceleration
   - 0.1377732103328372 x2_input_acceleration
   - 0.1714406890541802 x3_input_acceleration >= -0.085621064829509
 Relaxed_CLF_constraint: - 0.0591191580959986 x1_input_acceleration
   + 0.0876985412607359 x2_input_acceleration
   - 0.0798597602680569 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2577528014094418 x1_input_acceleration
   - 0.1391626707789693 x2_input_acceleration
   - 0.1708810548681549 x3_input_acceleration >= -0.0187507726214011
 CBF_Constraint_h_left: 0.1222471985905582 x1_input_acceleration
   - 0.1391626707789693 x2_input_acceleration
   - 0.1708810548681549 x3_input_acceleration >= -0.00587774035360713
 CBF_Constraint_h_left: - 0.5777528014094419 x1_input_acceleration
   - 0.1391626707789693 x2_input_acceleration
   - 0.1708810548681549 x3_input_acceleration >= -0.0855912208469118
 Relaxed_CLF_constraint: - 0.058977490258554 x1_input_acceleration
   + 0.0870115596414907 x2_input_acceleration
   - 0.0795602140376235 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2574540354754001 x1_input_acceleration
   - 0.1405372125022186 x2_input_acceleration
   - 0.1703103600547704 x3_input_acceleration >= -0.018759726805612
 CBF_Constraint_h_left: 0.1225459645245999 x1_input_acceleration
   - 0.1405372125022186 x2_input_acceleration
   - 0.1703103600547704 x3_input_acceleration >= -0.00594346
 CBF_Constraint_h_left: - 0.5774540354754001 x1_input_acceleration
   - 0.1405372125022186 x2_input_acceleration
   - 0.1703103600547704 x3_input_acceleration >= -0.085552372481676
 Relaxed_CLF_constraint: - 0.058827868872954 x1_input_acceleration
   + 0.086323752338063 x2_input_acceleration
   - 0.0792674756549792 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2571492755269871 x1_input_acceleration
   - 0.1419078885420162 x2_input_acceleration
   - 0.1696939477207008 x3_input_acceleration >= -0.0187649086568859
 CBF_Constraint_h_left: 0.1228507244730129 x1_input_acceleration
   - 0.1419078885420162 x2_input_acceleration
   - 0.1696939477207008 x3_input_acceleration >= -0.00600654630675834
 CBF_Constraint_h_left: - 0.5771492755269871 x1_input_acceleration
   - 0.1419078885420162 x2_input_acceleration
   - 0.1696939477207008 x3_input_acceleration >= -0.0855087927412038
 Relaxed_CLF_constraint: - 0.0586728662943774 x1_input_acceleration
   + 0.0856353744812807 x2_input_acceleration
   - 0.0789793562532337 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2568276397901495 x1_input_acceleration
   - 0.143300867077768 x2_input_acceleration
   - 0.1691643704103068 x3_input_acceleration >= -0.0187780398204336
 CBF_Constraint_h_left: 0.1231723602098505 x1_input_acceleration
   - 0.143300867077768 x2_input_acceleration
   - 0.1691643704103068 x3_input_acceleration >= -0.00608078826030519
 CBF_Constraint_h_left: - 0.5768276397901495 x1_input_acceleration
   - 0.143300867077768 x2_input_acceleration
   - 0.1691643704103068 x3_input_acceleration >= -0.0854704621868575
 Relaxed_CLF_constraint: - 0.0585139557745914 x1_input_acceleration
   + 0.0849465742125808 x2_input_acceleration
   - 0.0786946 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2565009050170337 x1_input_acceleration
   - 0.1446674461700654 x2_input_acceleration
   - 0.1685798011494219 x3_input_acceleration >= -0.0187851334028762
 CBF_Constraint_h_left: 0.1235033865173901 x1_input_acceleration
   - 0.1446718569137788 x2_input_acceleration
   - 0.1685998283100665 x3_input_acceleration >= -0.00615223369258216
 CBF_Constraint_h_left: - 0.5764966134826099 x1_input_acceleration
   - 0.1446718569137788 x2_input_acceleration
   - 0.1685998283100665 x3_input_acceleration >= -0.0854260484114956
 Relaxed_CLF_constraint: - 0.0583529725738246 x1_input_acceleration
   + 0.0842575626205336 x2_input_acceleration
   - 0.0784116137361795 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2561756328571393 x1_input_acceleration
   - 0.1460458185151821 x2_input_acceleration
   - 0.1680166646545068 x3_input_acceleration >= -0.0187962338942876
 CBF_Constraint_h_left: 0.1238243671428607 x1_input_acceleration
   - 0.1460458185151821 x2_input_acceleration
   - 0.1680166646545068 x3_input_acceleration >= -0.00622286365143111
 CBF_Constraint_h_left: - 0.5761756328571394 x1_input_acceleration
   - 0.1460458185151821 x2_input_acceleration
   - 0.1680166646545068 x3_input_acceleration >= -0.0853843351514299
 Relaxed_CLF_constraint: - 0.0581903364938774 x1_input_acceleration
   + 0.0835683764479753 x2_input_acceleration
   - 0.0781300454887219 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.255856287850871 x1_input_acceleration
   - 0.1474359866331942 x2_input_acceleration
   - 0.1674810300834572 x3_input_acceleration >= -0.0188124264062868
 CBF_Constraint_h_left: 0.124143712149129 x1_input_acceleration
   - 0.1474359866331942 x2_input_acceleration
   - 0.1674810300834572 x3_input_acceleration >= -0.00629973171462128
 CBF_Constraint_h_left: - 0.5758562878508711 x1_input_acceleration
   - 0.1474359866331942 x2_input_acceleration
   - 0.1674810300834572 x3_input_acceleration >= -0.0853494324624262
 Relaxed_CLF_constraint: - 0.0580266108748985 x1_input_acceleration
   + 0.0828790740395124 x2_input_acceleration
   - 0.0778494132045704 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2552612162948175 x1_input_acceleration
   - 0.1480649282300157 x2_input_acceleration
   - 0.1656866789956133 x3_input_acceleration >= -0.0186333967781662
 CBF_Constraint_h_left: 0.1247435520767646 x1_input_acceleration
   - 0.1480661203229112 x2_input_acceleration
   - 0.1656773806710283 x3_input_acceleration >= -0.00623338105955503
 CBF_Constraint_h_left: - 0.5752564479232354 x1_input_acceleration
   - 0.1480661203229112 x2_input_acceleration
   - 0.1656773806710283 x3_input_acceleration >= -0.0850731378326874
 Relaxed_CLF_constraint: - 0.0578621126640104 x1_input_acceleration
   + 0.0821896900254645 x2_input_acceleration
   - 0.0775694437572976 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2551833944933351 x1_input_acceleration
   - 0.1501937186086952 x2_input_acceleration
   - 0.166336761767469 x3_input_acceleration >= -0.0188361590624842
 CBF_Constraint_h_left: 0.1250688523633544 x1_input_acceleration
   - 0.1494437729681313 x2_input_acceleration
   - 0.1651276219435555 x3_input_acceleration >= -0.00631069765979267
 CBF_Constraint_h_left: - 0.5749311476366457 x1_input_acceleration
   - 0.1494437729681313 x2_input_acceleration
   - 0.1651276219435555 x3_input_acceleration >= -0.0850365993326186
 Relaxed_CLF_constraint: - 0.0576970783349245 x1_input_acceleration
   + 0.0815002676564067 x2_input_acceleration
   - 0.0772899152312687 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2545573258848711 x1_input_acceleration
   - 0.1507699944886285 x2_input_acceleration
   - 0.1645405360723056 x3_input_acceleration >= -0.0186511528526798
 CBF_Constraint_h_left: 0.1254426741151289 x1_input_acceleration
   - 0.1507699944886285 x2_input_acceleration
   - 0.1645405360723056 x3_input_acceleration >= -0.0063852609345543
 CBF_Constraint_h_left: - 0.5745573258848711 x1_input_acceleration
   - 0.1507699944886285 x2_input_acceleration
   - 0.1645405360723056 x3_input_acceleration >= -0.0849803249942592
 Relaxed_CLF_constraint: - 0.0575139821563028 x1_input_acceleration
   + 0.0808399583144626 x2_input_acceleration
   - 0.0769935708773393 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2541915802963891 x1_input_acceleration
   - 0.1520909056124149 x2_input_acceleration
   - 0.163943275976369 x3_input_acceleration >= -0.018655600200361
 CBF_Constraint_h_left: 0.1258084197036109 x1_input_acceleration
   - 0.1520909056124149 x2_input_acceleration
   - 0.163943275976369 x3_input_acceleration >= -0.0064592
 CBF_Constraint_h_left: - 0.5741915802963891 x1_input_acceleration
   - 0.1520909056124149 x2_input_acceleration
   - 0.163943275976369 x3_input_acceleration >= -0.0849262530477832
 Relaxed_CLF_constraint: - 0.0573306325249036 x1_input_acceleration
   + 0.0801796219618589 x2_input_acceleration
   - 0.0766974442244515 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.253829314165393 x1_input_acceleration
   - 0.153411732459563 x2_input_acceleration
   - 0.1633545284326999 x3_input_acceleration >= -0.018662295586347
 CBF_Constraint_h_left: 0.126170685834607 x1_input_acceleration
   - 0.153411732459563 x2_input_acceleration
   - 0.1633545284326999 x3_input_acceleration >= -0.00653472589492235
 CBF_Constraint_h_left: - 0.573829314165393 x1_input_acceleration
   - 0.153411732459563 x2_input_acceleration
   - 0.1633545284326999 x3_input_acceleration >= -0.0848749858528099
 Relaxed_CLF_constraint: - 0.0571471152736145 x1_input_acceleration
   + 0.0795192681429296 x2_input_acceleration
   - 0.0764014611272081 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.253726696353864 x1_input_acceleration
   - 0.155467508719755 x2_input_acceleration
   - 0.1639794078687086 x3_input_acceleration >= -0.0188591572287863
 CBF_Constraint_h_left: 0.1265379482689387 x1_input_acceleration
   - 0.1547324642403849 x2_input_acceleration
   - 0.1627627578595533 x3_input_acceleration >= -0.00661142579701554
 CBF_Constraint_h_left: - 0.5734620517310614 x1_input_acceleration
   - 0.1547324642403849 x2_input_acceleration
   - 0.1627627578595533 x3_input_acceleration >= -0.084823143902887
 Relaxed_CLF_constraint: - 0.0569634840564487 x1_input_acceleration
   + 0.0788589022525187 x2_input_acceleration
   - 0.0761055758406348 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2531070330309195 x1_input_acceleration
   - 0.1560441523565452 x2_input_acceleration
   - 0.1621770798704207 x3_input_acceleration >= -0.0186785882224211
 CBF_Constraint_h_left: 0.1268929669690805 x1_input_acceleration
   - 0.1560441523565452 x2_input_acceleration
   - 0.1621770798704207 x3_input_acceleration >= -0.00668825194654642
 CBF_Constraint_h_left: - 0.5731070330309195 x1_input_acceleration
   - 0.1560441523565452 x2_input_acceleration
   - 0.1621770798704207 x3_input_acceleration >= -0.0847757135073682
 Relaxed_CLF_constraint: - 0.0567797758233211 x1_input_acceleration
   + 0.0781985282414356 x2_input_acceleration
   - 0.0758097566138297 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.252680501333111 x1_input_acceleration
   - 0.1572796696921335 x2_input_acceleration
   - 0.1615252589086502 x3_input_acceleration >= -0.0186686848794814
 CBF_Constraint_h_left: 0.127319498666889 x1_input_acceleration
   - 0.1572796696921335 x2_input_acceleration
   - 0.1615252589086502 x3_input_acceleration >= -0.00675938962619026
 CBF_Constraint_h_left: - 0.5726805013331111 x1_input_acceleration
   - 0.1572796696921335 x2_input_acceleration
   - 0.1615252589086502 x3_input_acceleration >= -0.0846975650927791
 Relaxed_CLF_constraint: - 0.0565693709973661 x1_input_acceleration
   + 0.0775827961315638 x2_input_acceleration
   - 0.0754884356905922 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.25251467532823 x1_input_acceleration
   - 0.1592433876350638 x2_input_acceleration
   - 0.1620959197415546 x3_input_acceleration >= -0.0188493012396183
 CBF_Constraint_h_left: 0.12748532467177 x1_input_acceleration
   - 0.1592433876350638 x2_input_acceleration
   - 0.1620959197415546 x3_input_acceleration >= -0.00697151292725458
 CBF_Constraint_h_left: - 0.57251467532823 x1_input_acceleration
   - 0.1592433876350638 x2_input_acceleration
   - 0.1620959197415546 x3_input_acceleration >= -0.0848516492921351
 Relaxed_CLF_constraint: - 0.0563562814507362 x1_input_acceleration
   + 0.0769714396320225 x2_input_acceleration
   - 0.0751646662420847 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2520746802887744 x1_input_acceleration
   - 0.1604747910629436 x2_input_acceleration
   - 0.1614644527556265 x3_input_acceleration >= -0.0188411431282643
 CBF_Constraint_h_left: 0.1279253197112256 x1_input_acceleration
   - 0.1604747910629436 x2_input_acceleration
   - 0.1614644527556265 x3_input_acceleration >= -0.00704695387339715
 CBF_Constraint_h_left: - 0.5720746802887744 x1_input_acceleration
   - 0.1604747910629436 x2_input_acceleration
   - 0.1614644527556265 x3_input_acceleration >= -0.0847730919744682
 Relaxed_CLF_constraint: - 0.0561431980698023 x1_input_acceleration
   + 0.0763600890571512 x2_input_acceleration
   - 0.0748408861220759 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2516483527372266 x1_input_acceleration
   - 0.1616975155363571 x2_input_acceleration
   - 0.16081703698122 x3_input_acceleration >= -0.0188337748373523
 CBF_Constraint_h_left: 0.1283516472627734 x1_input_acceleration
   - 0.1616975155363571 x2_input_acceleration
   - 0.16081703698122 x3_input_acceleration >= -0.00712058781727924
 CBF_Constraint_h_left: - 0.5716483527372267 x1_input_acceleration
   - 0.1616975155363571 x2_input_acceleration
   - 0.16081703698122 x3_input_acceleration >= -0.0846975112753086
 Relaxed_CLF_constraint: - 0.0559300342940284 x1_input_acceleration
   + 0.0757487268204444 x2_input_acceleration
   - 0.0745171782348727 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2512353590127642 x1_input_acceleration
   - 0.1629099923593789 x2_input_acceleration
   - 0.1601668970245682 x3_input_acceleration >= -0.018828076532821
 CBF_Constraint_h_left: 0.1287646409872358 x1_input_acceleration
   - 0.1629099923593789 x2_input_acceleration
   - 0.1601668970245682 x3_input_acceleration >= -0.0071933583203958
 CBF_Constraint_h_left: - 0.5712353590127642 x1_input_acceleration
   - 0.1629099923593789 x2_input_acceleration
   - 0.1601668970245682 x3_input_acceleration >= -0.0846257339748633
 Relaxed_CLF_constraint: - 0.0557168617115824 x1_input_acceleration
   + 0.0751373624094829 x2_input_acceleration
   - 0.0741934791633766 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2505447968258426 x1_input_acceleration
   - 0.1633974426746021 x2_input_acceleration
   - 0.1582853665658745 x3_input_acceleration >= -0.018631419189499
 CBF_Constraint_h_left: 0.1294552031741574 x1_input_acceleration
   - 0.1633974426746021 x2_input_acceleration
   - 0.1582853665658745 x3_input_acceleration >= -0.00712790779258888
 CBF_Constraint_h_left: - 0.5705447968258426 x1_input_acceleration
   - 0.1633974426746021 x2_input_acceleration
   - 0.1582853665658745 x3_input_acceleration >= -0.0843185866816338
 Relaxed_CLF_constraint: - 0.0555036645109439 x1_input_acceleration
   + 0.0745259958028966 x2_input_acceleration
   - 0.0738698008213894 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2503923422864224 x1_input_acceleration
   - 0.1653668697960462 x2_input_acceleration
   - 0.1588616020829905 x3_input_acceleration >= -0.0188198838295495
 CBF_Constraint_h_left: 0.1296076577135776 x1_input_acceleration
   - 0.1653668697960462 x2_input_acceleration
   - 0.1588616020829905 x3_input_acceleration >= -0.0073453387951292
 CBF_Constraint_h_left: - 0.5703923422864224 x1_input_acceleration
   - 0.1653668697960462 x2_input_acceleration
   - 0.1588616020829905 x3_input_acceleration >= -0.084482658595377
 Relaxed_CLF_constraint: - 0.0552905849768294 x1_input_acceleration
   + 0.0739146457370476 x2_input_acceleration
   - 0.0735460172966832 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2496824247607293 x1_input_acceleration
   - 0.1658389132823479 x2_input_acceleration
   - 0.1569988501403167 x3_input_acceleration >= -0.0186231243346122
 CBF_Constraint_h_left: 0.1303175752392707 x1_input_acceleration
   - 0.1658389132823479 x2_input_acceleration
   - 0.1569988501403167 x3_input_acceleration >= -0.00728346363007367
 CBF_Constraint_h_left: - 0.5696886236437859 x1_input_acceleration
   - 0.1658389132823479 x2_input_acceleration
   - 0.1569862139556243 x3_input_acceleration >= -0.0841730861099856
 Relaxed_CLF_constraint: - 0.0550774852685483 x1_input_acceleration
   + 0.0733032935457462 x2_input_acceleration
   - 0.0732222510741866 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2492561483278371 x1_input_acceleration
   - 0.1670616260900644 x2_input_acceleration
   - 0.1563513841043147 x3_input_acceleration >= -0.018620992425605
 CBF_Constraint_h_left: 0.1307438516721629 x1_input_acceleration
   - 0.1670616260900644 x2_input_acceleration
   - 0.1563513841043147 x3_input_acceleration >= -0.00736232424331599
 CBF_Constraint_h_left: - 0.5692561483278371 x1_input_acceleration
   - 0.1670616260900644 x2_input_acceleration
   - 0.1563513841043147 x3_input_acceleration >= -0.084101976158059
 Relaxed_CLF_constraint: - 0.0548643470521022 x1_input_acceleration
   + 0.0726919371418879 x2_input_acceleration
   - 0.0728985180561857 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2488194114815665 x1_input_acceleration
   - 0.1682863619401089 x2_input_acceleration
   - 0.1556806461293954 x3_input_acceleration >= -0.0186169656810841
 CBF_Constraint_h_left: 0.1311805885184335 x1_input_acceleration
   - 0.1682863619401089 x2_input_acceleration
   - 0.1556806461293954 x3_input_acceleration >= -0.00744127749958649
 CBF_Constraint_h_left: - 0.5688194114815666 x1_input_acceleration
   - 0.1682863619401089 x2_input_acceleration
   - 0.1556806461293954 x3_input_acceleration >= -0.0840280715181348
 Relaxed_CLF_constraint: - 0.0546512238377072 x1_input_acceleration
   + 0.0720805824958268 x2_input_acceleration
   - 0.0725747719744572 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2486906686186946 x1_input_acceleration
   - 0.1702405440312422 x2_input_acceleration
   - 0.1562665300852143 x3_input_acceleration >= -0.018812029978735
 CBF_Constraint_h_left: 0.1313093313813054 x1_input_acceleration
   - 0.1702405440312422 x2_input_acceleration
   - 0.1562665300852143 x3_input_acceleration >= -0.007660802941183
 CBF_Constraint_h_left: - 0.5686906686186946 x1_input_acceleration
   - 0.1702405440312422 x2_input_acceleration
   - 0.1562665300852143 x3_input_acceleration >= -0.0842025369577261
 Relaxed_CLF_constraint: - 0.0544380792129118 x1_input_acceleration
   + 0.071469225550602 x2_input_acceleration
   - 0.0722510442975205 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2479811617552943 x1_input_acceleration
   - 0.1707249923202008 x2_input_acceleration
   - 0.1544168287770096 x3_input_acceleration >= -0.0186215591494466
 CBF_Constraint_h_left: 0.1320188382447057 x1_input_acceleration
   - 0.1707249923202008 x2_input_acceleration
   - 0.1544168287770096 x3_input_acceleration >= -0.00760513841594067
 CBF_Constraint_h_left: - 0.5679811617552943 x1_input_acceleration
   - 0.1707249923202008 x2_input_acceleration
   - 0.1544168287770096 x3_input_acceleration >= -0.0838985450302937
 Relaxed_CLF_constraint: - 0.054224946417198 x1_input_acceleration
   + 0.0708578698410287 x2_input_acceleration
   - 0.0719273064859779 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2478180979292989 x1_input_acceleration
   - 0.1726891867779263 x2_input_acceleration
   - 0.1549640791984659 x3_input_acceleration >= -0.0188123076832864
 CBF_Constraint_h_left: 0.1321819020707011 x1_input_acceleration
   - 0.1726891867779263 x2_input_acceleration
   - 0.1549640791984659 x3_input_acceleration >= -0.00782686907671959
 CBF_Constraint_h_left: - 0.5678180979292989 x1_input_acceleration
   - 0.1726891867779263 x2_input_acceleration
   - 0.1549640791984659 x3_input_acceleration >= -0.0840632033519742
 Relaxed_CLF_constraint: - 0.0540118074485362 x1_input_acceleration
   + 0.0702465135026689 x2_input_acceleration
   - 0.0716035739467671 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2471170912689903 x1_input_acceleration
   - 0.1731765043626219 x2_input_acceleration
   - 0.1531229124096154 x3_input_acceleration >= -0.0186258961913266
 CBF_Constraint_h_left: 0.1326082505278847 x1_input_acceleration
   - 0.1739119064698607 x2_input_acceleration
   - 0.1543166742351769 x3_input_acceleration >= -0.00791098381676338
 CBF_Constraint_h_left: - 0.5673917494721152 x1_input_acceleration
   - 0.1739119064698607 x2_input_acceleration
   - 0.1543166742351769 x3_input_acceleration >= -0.0839980961320037
 Relaxed_CLF_constraint: - 0.0537986332199444 x1_input_acceleration
   + 0.0696351536567017 x2_input_acceleration
   - 0.0712798714651226 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2466944617099935 x1_input_acceleration
   - 0.1743997117415788 x2_input_acceleration
   - 0.1524692722365633 x3_input_acceleration >= -0.0186300739675691
 CBF_Constraint_h_left: 0.1333093529872721 x1_input_acceleration
   - 0.1743992349044206 x2_input_acceleration
   - 0.1524755903289095 x3_input_acceleration >= -0.00785882059382045
 CBF_Constraint_h_left: - 0.566690647012728 x1_input_acceleration
   - 0.1743992349044206 x2_input_acceleration
   - 0.1524755903289095 x3_input_acceleration >= -0.0837005470482752
 Relaxed_CLF_constraint: - 0.0535854110918133 x1_input_acceleration
   + 0.0690237883858024 x2_input_acceleration
   - 0.0709562104247697 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2462773020264089 x1_input_acceleration
   - 0.1756224182448221 x2_input_acceleration
   - 0.1518165118802125 x3_input_acceleration >= -0.0186359991407602
 CBF_Constraint_h_left: 0.1337226979735911 x1_input_acceleration
   - 0.1756224182448221 x2_input_acceleration
   - 0.1518165118802125 x3_input_acceleration >= -0.00794331175574255
 CBF_Constraint_h_left: - 0.5662773020264089 x1_input_acceleration
   - 0.1756224182448221 x2_input_acceleration
   - 0.1518165118802125 x3_input_acceleration >= -0.0836403674649857
 Relaxed_CLF_constraint: - 0.053372301297018 x1_input_acceleration
   + 0.0684124351341807 x2_input_acceleration
   - 0.0706324528509644 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2460980188799455 x1_input_acceleration
   - 0.1775742148169064 x2_input_acceleration
   - 0.1523804567067269 x3_input_acceleration >= -0.0188291600626564
 CBF_Constraint_h_left: 0.1339019811200546 x1_input_acceleration
   - 0.1775742148169064 x2_input_acceleration
   - 0.1523804567067269 x3_input_acceleration >= -0.00817053647546673
 CBF_Constraint_h_left: - 0.5660980188799455 x1_input_acceleration
   - 0.1775742148169064 x2_input_acceleration
   - 0.1523804567067269 x3_input_acceleration >= -0.0838048430834477
 Relaxed_CLF_constraint: - 0.0531591588998117 x1_input_acceleration
   + 0.0678010788555848 x2_input_acceleration
   - 0.0703087228640141 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2457112255482223 x1_input_acceleration
   - 0.1788130234063166 x2_input_acceleration
   - 0.1517465353048127 x3_input_acceleration >= -0.018843778669283
 CBF_Constraint_h_left: 0.1343188151927445 x1_input_acceleration
   - 0.1787965725243585 x2_input_acceleration
   - 0.1517346143758577 x3_input_acceleration >= -0.00825828791525777
 CBF_Constraint_h_left: - 0.5656811848072555 x1_input_acceleration
   - 0.1787965725243585 x2_input_acceleration
   - 0.1517346143758577 x3_input_acceleration >= -0.0837467025977972
 Relaxed_CLF_constraint: - 0.0529459734918847 x1_input_acceleration
   + 0.0671897211879244 x2_input_acceleration
   - 0.0699850268381974 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2452846273137001 x1_input_acceleration
   - 0.1800357660002836 x2_input_acceleration
   - 0.1510993414539229 x3_input_acceleration >= -0.0188521091058847
 CBF_Constraint_h_left: 0.1347153726862999 x1_input_acceleration
   - 0.1800357660002836 x2_input_acceleration
   - 0.1510993414539229 x3_input_acceleration >= -0.00834803
 CBF_Constraint_h_left: - 0.5652846273137001 x1_input_acceleration
   - 0.1800357660002836 x2_input_acceleration
   - 0.1510993414539229 x3_input_acceleration >= -0.0836976494760767
 Relaxed_CLF_constraint: - 0.0527326743746236 x1_input_acceleration
   + 0.0665783498909409 x2_input_acceleration
   - 0.0696614299127524 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2448373671091797 x1_input_acceleration
   - 0.181254660400535 x2_input_acceleration
   - 0.150433515944467 x3_input_acceleration >= -0.0188572077423207
 CBF_Constraint_h_left: 0.1351626328908203 x1_input_acceleration
   - 0.181254660400535 x2_input_acceleration
   - 0.150433515944467 x3_input_acceleration >= -0.00843810799157653
 CBF_Constraint_h_left: - 0.5648373671091798 x1_input_acceleration
   - 0.181254660400535 x2_input_acceleration
   - 0.150433515944467 x3_input_acceleration >= -0.0836311864797894
 Relaxed_CLF_constraint: - 0.0525195346898438 x1_input_acceleration
   + 0.0659669953421824 x2_input_acceleration
   - 0.0693376962733199 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2443995302706795 x1_input_acceleration
   - 0.1824785761473382 x2_input_acceleration
   - 0.1498071598651811 x3_input_acceleration >= -0.0188679365740402
 CBF_Constraint_h_left: 0.1356004697293205 x1_input_acceleration
   - 0.1824785761473382 x2_input_acceleration
   - 0.1498071598651811 x3_input_acceleration >= -0.00853202582261105
 CBF_Constraint_h_left: - 0.5643995302706795 x1_input_acceleration
   - 0.1824785761473382 x2_input_acceleration
   - 0.1498071598651811 x3_input_acceleration >= -0.0835718614173488
 Relaxed_CLF_constraint: - 0.0523063383164922 x1_input_acceleration
   + 0.0653558719338077 x2_input_acceleration
   - 0.0690137893976174 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.243661669088756 x1_input_acceleration
   - 0.1828858435808283 x2_input_acceleration
   - 0.1479004957560263 x3_input_acceleration >= -0.018673199352567
 CBF_Constraint_h_left: 0.136063672708119 x1_input_acceleration
   - 0.183634001082049 x2_input_acceleration
   - 0.1491247751597128 x3_input_acceleration >= -0.00861824198766475
 CBF_Constraint_h_left: - 0.5639363272918809 x1_input_acceleration
   - 0.183634001082049 x2_input_acceleration
   - 0.1491247751597128 x3_input_acceleration >= -0.0834959565398231
 Relaxed_CLF_constraint: - 0.0520747368270929 x1_input_acceleration
   + 0.0647781594664523 x2_input_acceleration
   - 0.0686725970448833 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2431759876432217 x1_input_acceleration
   - 0.1840005262798805 x2_input_acceleration
   - 0.1471970590524695 x3_input_acceleration >= -0.0186644322078064
 CBF_Constraint_h_left: 0.1368240123567783 x1_input_acceleration
   - 0.1840005262798805 x2_input_acceleration
   - 0.1471970590524695 x3_input_acceleration >= -0.00856099455559425
 CBF_Constraint_h_left: - 0.5631759876432216 x1_input_acceleration
   - 0.1840005262798805 x2_input_acceleration
   - 0.1471970590524695 x3_input_acceleration >= -0.0831725902307218
 Relaxed_CLF_constraint: - 0.0518318961043258 x1_input_acceleration
   + 0.0642208181169261 x2_input_acceleration
   - 0.0683208786931048 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2427132261873408 x1_input_acceleration
   - 0.1851131697965804 x2_input_acceleration
   - 0.1464958513663609 x3_input_acceleration >= -0.0186594075664899
 CBF_Constraint_h_left: 0.1372867738126592 x1_input_acceleration
   - 0.1851131697965804 x2_input_acceleration
   - 0.1464958513663609 x3_input_acceleration >= -0.00864389459089519
 CBF_Constraint_h_left: - 0.5627132261873409 x1_input_acceleration
   - 0.1851131697965804 x2_input_acceleration
   - 0.1464958513663609 x3_input_acceleration >= -0.0830935237564645
 Relaxed_CLF_constraint: - 0.0515890712845885 x1_input_acceleration
   + 0.063663483079615 x2_input_acceleration
   - 0.0679691423617998 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2422208685543447 x1_input_acceleration
   - 0.1862335599669167 x2_input_acceleration
   - 0.1457808381671599 x3_input_acceleration >= -0.018651485199473
 CBF_Constraint_h_left: 0.1377791314456553 x1_input_acceleration
   - 0.1862335599669167 x2_input_acceleration
   - 0.1457808381671599 x3_input_acceleration >= -0.0087295201741475
 CBF_Constraint_h_left: - 0.5622208685543446 x1_input_acceleration
   - 0.1862335599669167 x2_input_acceleration
   - 0.1457808381671599 x3_input_acceleration >= -0.0830068241681681
 Relaxed_CLF_constraint: - 0.0513462303281978 x1_input_acceleration
   + 0.0631061490173961 x2_input_acceleration
   - 0.0676174174127425 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2420026531312809 x1_input_acceleration
   - 0.1880855439218101 x2_input_acceleration
   - 0.1462789343494662 x3_input_acceleration >= -0.0188347456473394
 CBF_Constraint_h_left: 0.1379973468687191 x1_input_acceleration
   - 0.1880855439218101 x2_input_acceleration
   - 0.1462789343494662 x3_input_acceleration >= -0.00895424155239604
 CBF_Constraint_h_left: - 0.561735147485529 x1_input_acceleration
   - 0.1873482344659385 x2_input_acceleration
   - 0.1450774239200839 x3_input_acceleration >= -0.0829232989523453
 Relaxed_CLF_constraint: - 0.0511033697937899 x1_input_acceleration
   + 0.0625488117678852 x2_input_acceleration
   - 0.0672657102892045 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2415143051158029 x1_input_acceleration
   - 0.1892240218336108 x2_input_acceleration
   - 0.1456046736559256 x3_input_acceleration >= -0.0188339027512261
 CBF_Constraint_h_left: 0.1384756813038749 x1_input_acceleration
   - 0.189204590719414 x2_input_acceleration
   - 0.1455996668657645 x3_input_acceleration >= -0.00904328861332371
 CBF_Constraint_h_left: - 0.5615243186961252 x1_input_acceleration
   - 0.189204590719414 x2_input_acceleration
   - 0.1455996668657645 x3_input_acceleration >= -0.0830768001569675
 Relaxed_CLF_constraint: - 0.0508601494603673 x1_input_acceleration
   + 0.0619916725548743 x2_input_acceleration
   - 0.0669140960137538 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2407670293678258 x1_input_acceleration
   - 0.1895727080119627 x2_input_acceleration
   - 0.1436854022239543 x3_input_acceleration >= -0.018638017216464
 CBF_Constraint_h_left: 0.1389716638694789 x1_input_acceleration
   - 0.1903377932322996 x2_input_acceleration
   - 0.144900621721635 x3_input_acceleration >= -0.00913444726662736
 CBF_Constraint_h_left: - 0.5610283361305211 x1_input_acceleration
   - 0.1903377932322996 x2_input_acceleration
   - 0.144900621721635 x3_input_acceleration >= -0.0829943649123097
 Relaxed_CLF_constraint: - 0.0506171649677264 x1_input_acceleration
   + 0.0614347868555298 x2_input_acceleration
   - 0.0665620700466085 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2402476145785651 x1_input_acceleration
   - 0.1906349120282627 x2_input_acceleration
   - 0.1429389219869746 x3_input_acceleration >= -0.0186230303533781
 CBF_Constraint_h_left: 0.1397519085842767 x1_input_acceleration
   - 0.1906259713315464 x2_input_acceleration
   - 0.1429510813345088 x3_input_acceleration >= -0.00907596713843723
 CBF_Constraint_h_left: - 0.5602480914157233 x1_input_acceleration
   - 0.1906259713315464 x2_input_acceleration
   - 0.1429510813345088 x3_input_acceleration >= -0.0826627991339404
 Relaxed_CLF_constraint: - 0.0503576959916752 x1_input_acceleration
   + 0.060908155195738 x2_input_acceleration
   - 0.0661949096018857 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.239989404293462 x1_input_acceleration
   - 0.1924331230430503 x2_input_acceleration
   - 0.1434319926807224 x3_input_acceleration >= -0.0187995393853988
 CBF_Constraint_h_left: 0.140010595706538 x1_input_acceleration
   - 0.1924331230430503 x2_input_acceleration
   - 0.1434319926807224 x3_input_acceleration >= -0.00930155256964104
 CBF_Constraint_h_left: - 0.559989404293462 x1_input_acceleration
   - 0.1924331230430503 x2_input_acceleration
   - 0.1434319926807224 x3_input_acceleration >= -0.0827978440723527
 Relaxed_CLF_constraint: - 0.050097937467776 x1_input_acceleration
   + 0.0603817575321247 x2_input_acceleration
   - 0.0658277555261522 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2394633991191882 x1_input_acceleration
   - 0.193490239265667 x2_input_acceleration
   - 0.1426836263277148 x3_input_acceleration >= -0.0187849523577069
 CBF_Constraint_h_left: 0.1405304019977552 x1_input_acceleration
   - 0.1934859477312432 x2_input_acceleration
   - 0.1426979314424609 x3_input_acceleration >= -0.00938707637326633
 CBF_Constraint_h_left: - 0.5594695980022448 x1_input_acceleration
   - 0.1934859477312432 x2_input_acceleration
   - 0.1426979314424609 x3_input_acceleration >= -0.082701435674052
 Relaxed_CLF_constraint: - 0.0498380343221674 x1_input_acceleration
   + 0.0598553451880283 x2_input_acceleration
   - 0.0654607249070214 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2389451900867706 x1_input_acceleration
   - 0.1945348687939337 x2_input_acceleration
   - 0.1419528091220716 x3_input_acceleration >= -0.0187723047649808
 CBF_Constraint_h_left: 0.1410548099132294 x1_input_acceleration
   - 0.1945348687939337 x2_input_acceleration
   - 0.1419528091220716 x3_input_acceleration >= -0.00947271864849438
 CBF_Constraint_h_left: - 0.5589451900867706 x1_input_acceleration
   - 0.1945348687939337 x2_input_acceleration
   - 0.1419528091220716 x3_input_acceleration >= -0.0826035351788641
 Relaxed_CLF_constraint: - 0.0495779761316422 x1_input_acceleration
   + 0.0593289177034054 x2_input_acceleration
   - 0.0650938261880804 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2384214612960153 x1_input_acceleration
   - 0.1955783809550726 x2_input_acceleration
   - 0.141216075858171 x3_input_acceleration >= -0.0187594190960789
 CBF_Constraint_h_left: 0.1415785387039847 x1_input_acceleration
   - 0.1955783809550726 x2_input_acceleration
   - 0.141216075858171 x3_input_acceleration >= -0.00955934144983597
 CBF_Constraint_h_left: - 0.5584214612960153 x1_input_acceleration
   - 0.1955783809550726 x2_input_acceleration
   - 0.141216075858171 x3_input_acceleration >= -0.0825068529034413
 Relaxed_CLF_constraint: - 0.0493175422477392 x1_input_acceleration
   + 0.0588024528558987 x2_input_acceleration
   - 0.0647272476954734 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2376456909399234 x1_input_acceleration
   - 0.1959015440332517 x2_input_acceleration
   - 0.1392627168042962 x3_input_acceleration >= -0.0185617484171598
 CBF_Constraint_h_left: 0.1423543090600767 x1_input_acceleration
   - 0.1959015440332517 x2_input_acceleration
   - 0.1392627168042962 x3_input_acceleration >= -0.00950906713857437
 CBF_Constraint_h_left: - 0.5576456909399233 x1_input_acceleration
   - 0.1959015440332517 x2_input_acceleration
   - 0.1392627168042962 x3_input_acceleration >= -0.0821850589675475
 Relaxed_CLF_constraint: - 0.0490574494280916 x1_input_acceleration
   + 0.0582761504954286 x2_input_acceleration
   - 0.0643602640799435 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.237399761199084 x1_input_acceleration
   - 0.1976909373696003 x2_input_acceleration
   - 0.139760247338203 x3_input_acceleration >= -0.0187433200178722
 CBF_Constraint_h_left: 0.142600238800916 x1_input_acceleration
   - 0.1976909373696003 x2_input_acceleration
   - 0.139760247338203 x3_input_acceleration >= -0.00973736539004629
 CBF_Constraint_h_left: - 0.557399761199084 x1_input_acceleration
   - 0.1976909373696003 x2_input_acceleration
   - 0.139760247338203 x3_input_acceleration >= -0.0823272818097257
 Relaxed_CLF_constraint: - 0.0487971554561095 x1_input_acceleration
   + 0.0577500489505453 x2_input_acceleration
   - 0.0639932537617223 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2368635105054007 x1_input_acceleration
   - 0.1987162294030902 x2_input_acceleration
   - 0.139027960175107 x3_input_acceleration >= -0.0187303090368937
 CBF_Constraint_h_left: 0.1431364894945993 x1_input_acceleration
   - 0.1987162294030902 x2_input_acceleration
   - 0.139027960175107 x3_input_acceleration >= -0.00982624204086756
 CBF_Constraint_h_left: - 0.5568635105054007 x1_input_acceleration
   - 0.1987162294030902 x2_input_acceleration
   - 0.139027960175107 x3_input_acceleration >= -0.0822284707177578
 Relaxed_CLF_constraint: - 0.0485311758764797 x1_input_acceleration
   + 0.0572349591433645 x2_input_acceleration
   - 0.0636207324831834 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2363299664201664 x1_input_acceleration
   - 0.1997392249721723 x2_input_acceleration
   - 0.1382628545950207 x3_input_acceleration >= -0.0187160569953512
 CBF_Constraint_h_left: 0.1436700335798336 x1_input_acceleration
   - 0.1997392249721723 x2_input_acceleration
   - 0.1382628545950207 x3_input_acceleration >= -0.0099133633755196
 CBF_Constraint_h_left: - 0.5563299664201664 x1_input_acceleration
   - 0.1997392249721723 x2_input_acceleration
   - 0.1382628545950207 x3_input_acceleration >= -0.0821288516225779
 Relaxed_CLF_constraint: - 0.048265357508179 x1_input_acceleration
   + 0.0567198850801369 x2_input_acceleration
   - 0.0632480740641729 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2357854835409408 x1_input_acceleration
   - 0.2007885088733155 x2_input_acceleration
   - 0.1375123266633682 x3_input_acceleration >= -0.0187051148821444
 CBF_Constraint_h_left: 0.1444844062906022 x1_input_acceleration
   - 0.2000392784884888 x2_input_acceleration
   - 0.1363128427919082 x3_input_acceleration >= -0.00986816192733861
 CBF_Constraint_h_left: - 0.5555155937093978 x1_input_acceleration
   - 0.2000392784884888 x2_input_acceleration
   - 0.1363128427919082 x3_input_acceleration >= -0.0817986197256279
 Relaxed_CLF_constraint: - 0.0479995533702019 x1_input_acceleration
   + 0.0562047798727294 x2_input_acceleration
   - 0.0628754327027168 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2349891673400901 x1_input_acceleration
   - 0.2010715229714419 x2_input_acceleration
   - 0.135554508180263 x3_input_acceleration >= -0.0185061727013092
 CBF_Constraint_h_left: 0.1450108326599099 x1_input_acceleration
   - 0.2010715229714419 x2_input_acceleration
   - 0.135554508180263 x3_input_acceleration >= -0.00995823090669212
 CBF_Constraint_h_left: - 0.5549891673400902 x1_input_acceleration
   - 0.2010715229714419 x2_input_acceleration
   - 0.135554508180263 x3_input_acceleration >= -0.0817044394757237
 Relaxed_CLF_constraint: - 0.047733955999757 x1_input_acceleration
   + 0.0556897305148588 x2_input_acceleration
   - 0.0625025834892404 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2344728292037705 x1_input_acceleration
   - 0.2021018521925193 x2_input_acceleration
   - 0.1348063615045659 x3_input_acceleration >= -0.0184988553491418
 CBF_Constraint_h_left: 0.1455271707962295 x1_input_acceleration
   - 0.2021018521925193 x2_input_acceleration
   - 0.1348063615045659 x3_input_acceleration >= -0.0100490178004254
 CBF_Constraint_h_left: - 0.5544728292037705 x1_input_acceleration
   - 0.2021018521925193 x2_input_acceleration
   - 0.1348063615045659 x3_input_acceleration >= -0.0816145080217451
 Relaxed_CLF_constraint: - 0.0474683959556451 x1_input_acceleration
   + 0.0551746851136097 x2_input_acceleration
   - 0.0621297022442873 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2342068827458737 x1_input_acceleration
   - 0.2038532728773852 x2_input_acceleration
   - 0.1352689788159361 x3_input_acceleration >= -0.0186766793545668
 CBF_Constraint_h_left: 0.1457931172541263 x1_input_acceleration
   - 0.2038532728773852 x2_input_acceleration
   - 0.1352689788159361 x3_input_acceleration >= -0.0102773716328508
 CBF_Constraint_h_left: - 0.5542068827458737 x1_input_acceleration
   - 0.2038532728773852 x2_input_acceleration
   - 0.1352689788159361 x3_input_acceleration >= -0.0817497805939066
 Relaxed_CLF_constraint: - 0.0472028619967162 x1_input_acceleration
   + 0.0546596424767126 x2_input_acceleration
   - 0.0617567986158843 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2336743803162074 x1_input_acceleration
   - 0.2048950354799823 x2_input_acceleration
   - 0.1345371176179998 x3_input_acceleration >= -0.0186714818993666
 CBF_Constraint_h_left: 0.1463256196837926 x1_input_acceleration
   - 0.2048950354799823 x2_input_acceleration
   - 0.1345371176179998 x3_input_acceleration >= -0.0103733496392872
 CBF_Constraint_h_left: - 0.5536743803162074 x1_input_acceleration
   - 0.2048950354799823 x2_input_acceleration
   - 0.1345371176179998 x3_input_acceleration >= -0.0816593827499598
 Relaxed_CLF_constraint: - 0.0469373260376204 x1_input_acceleration
   + 0.054144602430602 x2_input_acceleration
   - 0.0613838942734774 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2331024598902938 x1_input_acceleration
   - 0.2058711795259143 x2_input_acceleration
   - 0.1337774865661825 x3_input_acceleration >= -0.0186540288195656
 CBF_Constraint_h_left: 0.1468975401097062 x1_input_acceleration
   - 0.2058711795259143 x2_input_acceleration
   - 0.1337774865661825 x3_input_acceleration >= -0.0104645614404098
 CBF_Constraint_h_left: - 0.5531024598902938 x1_input_acceleration
   - 0.2058711795259143 x2_input_acceleration
   - 0.1337774865661825 x3_input_acceleration >= -0.0815504224020126
 Relaxed_CLF_constraint: - 0.0466578031262993 x1_input_acceleration
   + 0.0536565900122808 x2_input_acceleration
   - 0.0609980586784464 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2322674269475611 x1_input_acceleration
   - 0.2060631198519514 x2_input_acceleration
   - 0.1317584945151827 x3_input_acceleration >= -0.0184426169652169
 CBF_Constraint_h_left: 0.1477325730524389 x1_input_acceleration
   - 0.2060631198519514 x2_input_acceleration
   - 0.1317584945151827 x3_input_acceleration >= -0.0104118058451803
 CBF_Constraint_h_left: - 0.5522674269475611 x1_input_acceleration
   - 0.2060631198519514 x2_input_acceleration
   - 0.1317584945151827 x3_input_acceleration >= -0.0812054052768267
 Relaxed_CLF_constraint: - 0.0463661716646986 x1_input_acceleration
   + 0.0531921439352608 x2_input_acceleration
   - 0.0606008811687241 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2319555915248869 x1_input_acceleration
   - 0.2077225786097008 x2_input_acceleration
   - 0.1321629998203795 x3_input_acceleration >= -0.0186047811563613
 CBF_Constraint_h_left: 0.1480596672641756 x1_input_acceleration
   - 0.2077262740976769 x2_input_acceleration
   - 0.1321810004231017 x3_input_acceleration >= -0.0106359217234334
 CBF_Constraint_h_left: - 0.5519403327358243 x1_input_acceleration
   - 0.2077262740976769 x2_input_acceleration
   - 0.1321810004231017 x3_input_acceleration >= -0.0813150381809719
 Relaxed_CLF_constraint: - 0.0460734016889572 x1_input_acceleration
   + 0.0527299367960712 x2_input_acceleration
   - 0.0602026170252105 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2313499189676119 x1_input_acceleration
   - 0.2086548712797058 x2_input_acceleration
   - 0.1313744144071416 x3_input_acceleration >= -0.0185797192689726
 CBF_Constraint_h_left: 0.1486500810323881 x1_input_acceleration
   - 0.2086548712797058 x2_input_acceleration
   - 0.1313744144071416 x3_input_acceleration >= -0.0107232346651264
 CBF_Constraint_h_left: - 0.5513499189676119 x1_input_acceleration
   - 0.2086548712797058 x2_input_acceleration
   - 0.1313744144071416 x3_input_acceleration >= -0.0811957063037905
 Relaxed_CLF_constraint: - 0.045780578990642 x1_input_acceleration
   + 0.0522676647629791 x2_input_acceleration
   - 0.0598044500166811 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2307501332217427 x1_input_acceleration
   - 0.2095871511494636 x2_input_acceleration
   - 0.1305839184453762 x3_input_acceleration >= -0.0185561394163372
 CBF_Constraint_h_left: 0.1492498667782573 x1_input_acceleration
   - 0.2095871511494636 x2_input_acceleration
   - 0.1305839184453762 x3_input_acceleration >= -0.0108136141042061
 CBF_Constraint_h_left: - 0.5507501332217427 x1_input_acceleration
   - 0.2095871511494636 x2_input_acceleration
   - 0.1305839184453762 x3_input_acceleration >= -0.081076160731816
 Relaxed_CLF_constraint: - 0.0454878386750804 x1_input_acceleration
   + 0.0518054587346554 x2_input_acceleration
   - 0.0594061621989149 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.230185138185378 x1_input_acceleration
   - 0.2104989241126602 x2_input_acceleration
   - 0.1297716187260341 x3_input_acceleration >= -0.0185339169801961
 CBF_Constraint_h_left: 0.149814861814622 x1_input_acceleration
   - 0.2104989241126602 x2_input_acceleration
   - 0.1297716187260341 x3_input_acceleration >= -0.0108987407249742
 CBF_Constraint_h_left: - 0.5501851381853781 x1_input_acceleration
   - 0.2104989241126602 x2_input_acceleration
   - 0.1297716187260341 x3_input_acceleration >= -0.0809635390898565
 Relaxed_CLF_constraint: - 0.0451950891579967 x1_input_acceleration
   + 0.0513432541607108 x2_input_acceleration
   - 0.0590078801523542 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2295877099817557 x1_input_acceleration
   - 0.2114189234428918 x2_input_acceleration
   - 0.128983287190219 x3_input_acceleration >= -0.0185112915347032
 CBF_Constraint_h_left: 0.1504122900182443 x1_input_acceleration
   - 0.2114189234428918 x2_input_acceleration
   - 0.128983287190219 x3_input_acceleration >= -0.0109896266381696
 CBF_Constraint_h_left: - 0.5495877099817558 x1_input_acceleration
   - 0.2114189234428918 x2_input_acceleration
   - 0.128983287190219 x3_input_acceleration >= -0.0808453251317841
 Relaxed_CLF_constraint: - 0.0449023355206631 x1_input_acceleration
   + 0.0508810491237384 x2_input_acceleration
   - 0.0586096016639571 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2287380112415991 x1_input_acceleration
   - 0.2116049581694492 x2_input_acceleration
   - 0.126957349055297 x3_input_acceleration >= -0.0183039761469512
 CBF_Constraint_h_left: 0.1512619887584009 x1_input_acceleration
   - 0.2116049581694492 x2_input_acceleration
   - 0.126957349055297 x3_input_acceleration >= -0.0109437540110474
 CBF_Constraint_h_left: - 0.5490021790272437 x1_input_acceleration
   - 0.2123433405089268 x2_input_acceleration
   - 0.1281867544584342 x3_input_acceleration >= -0.0807312327134347
 Relaxed_CLF_constraint: - 0.0446095700434071 x1_input_acceleration
   + 0.0504188405907209 x2_input_acceleration
   - 0.0582113352980647 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2281548833934522 x1_input_acceleration
   - 0.2125416468470793 x2_input_acceleration
   - 0.1261749816669743 x3_input_acceleration >= -0.0182871821148524
 CBF_Constraint_h_left: 0.1518451166065478 x1_input_acceleration
   - 0.2125416468470793 x2_input_acceleration
   - 0.1261749816669743 x3_input_acceleration >= -0.0110377542700965
 CBF_Constraint_h_left: - 0.5481548833934522 x1_input_acceleration
   - 0.2125416468470793 x2_input_acceleration
   - 0.1261749816669743 x3_input_acceleration >= -0.0803919634578047
 Relaxed_CLF_constraint: - 0.0443168140264381 x1_input_acceleration
   + 0.0499566355303177 x2_input_acceleration
   - 0.0578130586511751 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2275629793031588 x1_input_acceleration
   - 0.2134592831327087 x2_input_acceleration
   - 0.125395045629089 x3_input_acceleration >= -0.0182684231432953
 CBF_Constraint_h_left: 0.1524308218137846 x1_input_acceleration
   - 0.2134660780622131 x2_input_acceleration
   - 0.125378594747131 x3_input_acceleration >= -0.0111306784857135
 CBF_Constraint_h_left: - 0.5475691781862154 x1_input_acceleration
   - 0.2134660780622131 x2_input_acceleration
   - 0.125378594747131 x3_input_acceleration >= -0.0802798908508889
 Relaxed_CLF_constraint: - 0.0440239614228197 x1_input_acceleration
   + 0.0494944199227508 x2_input_acceleration
   - 0.0574148651912535 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2270066061892622 x1_input_acceleration
   - 0.2143834492994872 x2_input_acceleration
   - 0.1245741291120608 x3_input_acceleration >= -0.0182527440577852
 CBF_Constraint_h_left: 0.1530224808773882 x1_input_acceleration
   - 0.2143836877180663 x2_input_acceleration
   - 0.1245984478071292 x3_input_acceleration >= -0.0112252546023535
 CBF_Constraint_h_left: - 0.5469775191226118 x1_input_acceleration
   - 0.2143836877180663 x2_input_acceleration
   - 0.1245984478071292 x3_input_acceleration >= -0.0801673862952676
 Relaxed_CLF_constraint: - 0.0437312313325462 x1_input_acceleration
   + 0.049032217630072 x2_input_acceleration
   - 0.0570165662802735 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2264001380908791 x1_input_acceleration
   - 0.2152933140170326 x2_input_acceleration
   - 0.1237628925153823 x3_input_acceleration >= -0.0182313717879449
 CBF_Constraint_h_left: 0.1536065376293358 x1_input_acceleration
   - 0.2152964134585609 x2_input_acceleration
   - 0.1237928140470595 x3_input_acceleration >= -0.0113180437150705
 CBF_Constraint_h_left: - 0.5463934623706642 x1_input_acceleration
   - 0.2152964134585609 x2_input_acceleration
   - 0.1237928140470595 x3_input_acceleration >= -0.080055755544803
 Relaxed_CLF_constraint: - 0.0434384877008351 x1_input_acceleration
   + 0.0485700135046367 x2_input_acceleration
   - 0.0566182793532416 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2260835748851214 x1_input_acceleration
   - 0.2169691011517156 x2_input_acceleration
   - 0.1242064645181531 x3_input_acceleration >= -0.0184041548788797
 CBF_Constraint_h_left: 0.1541853612721051 x1_input_acceleration
   - 0.2162177249996771 x2_input_acceleration
   - 0.1229663302789563 x3_input_acceleration >= -0.0114109871542297
 CBF_Constraint_h_left: - 0.5458146387278948 x1_input_acceleration
   - 0.2162177249996771 x2_input_acceleration
   - 0.1229663302789563 x3_input_acceleration >= -0.0799461107089929
 Relaxed_CLF_constraint: - 0.043145738019343 x1_input_acceleration
   + 0.0481078080133145 x2_input_acceleration
   - 0.0562199982350287 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2255099658547835 x1_input_acceleration
   - 0.217891130855191 x2_input_acceleration
   - 0.1234079016876447 x3_input_acceleration >= -0.0183902
 CBF_Constraint_h_left: 0.1544900341452165 x1_input_acceleration
   - 0.217891130855191 x2_input_acceleration
   - 0.1234079016876447 x3_input_acceleration >= -0.0116433064386229
 CBF_Constraint_h_left: - 0.5455099658547835 x1_input_acceleration
   - 0.217891130855191 x2_input_acceleration
   - 0.1234079016876447 x3_input_acceleration >= -0.0800717944877971
 Relaxed_CLF_constraint: - 0.0428529730396965 x1_input_acceleration
   + 0.0476456010686813 x2_input_acceleration
   - 0.0558217300987357 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2246396382826868 x1_input_acceleration
   - 0.2180867986703814 x2_input_acceleration
   - 0.1214076618535871 x3_input_acceleration >= -0.0181911597997067
 CBF_Constraint_h_left: 0.1553603617173132 x1_input_acceleration
   - 0.2180867986703814 x2_input_acceleration
   - 0.1214076618535871 x3_input_acceleration >= -0.0116096285259962
 CBF_Constraint_h_left: - 0.5446396382826868 x1_input_acceleration
   - 0.2180867986703814 x2_input_acceleration
   - 0.1214076618535871 x3_input_acceleration >= -0.0797335019249366
 Relaxed_CLF_constraint: - 0.0425599067267927 x1_input_acceleration
   + 0.0471833443629294 x2_input_acceleration
   - 0.0554237363032278 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2240762699487434 x1_input_acceleration
   - 0.2189958086020771 x2_input_acceleration
   - 0.1205943589378257 x3_input_acceleration >= -0.0181780845867612
 CBF_Constraint_h_left: 0.1559237300512566 x1_input_acceleration
   - 0.2189958086020771 x2_input_acceleration
   - 0.1205943589378257 x3_input_acceleration >= -0.0117035932965
 CBF_Constraint_h_left: - 0.5440762699487434 x1_input_acceleration
   - 0.2189958086020771 x2_input_acceleration
   - 0.1205943589378257 x3_input_acceleration >= -0.0796302877785602
 Relaxed_CLF_constraint: - 0.0422672553051823 x1_input_acceleration
   + 0.0467211503979055 x2_input_acceleration
   - 0.0550253698909709 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2237445694897461 x1_input_acceleration
   - 0.2206510766482979 x2_input_acceleration
   - 0.1210027771804866 x3_input_acceleration >= -0.0183475505218988
 CBF_Constraint_h_left: 0.1562554305102539 x1_input_acceleration
   - 0.2206510766482979 x2_input_acceleration
   - 0.1210027771804866 x3_input_acceleration >= -0.0119360823188471
 CBF_Constraint_h_left: - 0.5437445694897461 x1_input_acceleration
   - 0.2206510766482979 x2_input_acceleration
   - 0.1210027771804866 x3_input_acceleration >= -0.0797466816402582
 Relaxed_CLF_constraint: - 0.0419745663916017 x1_input_acceleration
   + 0.046258952451913 x2_input_acceleration
   - 0.054627035658267 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2231600892458839 x1_input_acceleration
   - 0.2215966991641689 x2_input_acceleration
   - 0.1202332186594927 x3_input_acceleration >= -0.0183403873454769
 CBF_Constraint_h_left: 0.1568399107541161 x1_input_acceleration
   - 0.2215966991641689 x2_input_acceleration
   - 0.1202332186594927 x3_input_acceleration >= -0.0120399703887589
 CBF_Constraint_h_left: - 0.5431600892458839 x1_input_acceleration
   - 0.2215966991641689 x2_input_acceleration
   - 0.1202332186594927 x3_input_acceleration >= -0.0796460016248183
 Relaxed_CLF_constraint: - 0.0416818494325124 x1_input_acceleration
   + 0.0457967508207475 x2_input_acceleration
   - 0.054228726143406 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2225722240161889 x1_input_acceleration
   - 0.2225102610864576 x2_input_acceleration
   - 0.1194338997105788 x3_input_acceleration >= -0.0183284168980882
 CBF_Constraint_h_left: 0.1574277759838111 x1_input_acceleration
   - 0.2225102610864576 x2_input_acceleration
   - 0.1194338997105788 x3_input_acceleration >= -0.0121396943350123
 CBF_Constraint_h_left: - 0.542572224016189 x1_input_acceleration
   - 0.2225102610864576 x2_input_acceleration
   - 0.1194338997105788 x3_input_acceleration >= -0.0795399727406784
 Relaxed_CLF_constraint: - 0.0413891089105604 x1_input_acceleration
   + 0.0453345458369286 x2_input_acceleration
   - 0.0538304375757789 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2220010554254237 x1_input_acceleration
   - 0.2234315677488958 x2_input_acceleration
   - 0.1186066767781117 x3_input_acceleration >= -0.0183184194632697
 CBF_Constraint_h_left: 0.1579979909002599 x1_input_acceleration
   - 0.2234406276549017 x2_input_acceleration
   - 0.1186125180332997 x3_input_acceleration >= -0.0122395021623837
 CBF_Constraint_h_left: - 0.5420020090997402 x1_input_acceleration
   - 0.2234406276549017 x2_input_acceleration
   - 0.1186125180332997 x3_input_acceleration >= -0.0794402053472928
 Relaxed_CLF_constraint: - 0.0410963720578047 x1_input_acceleration
   + 0.0448723427849453 x2_input_acceleration
   - 0.0534321445032526 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2214073004851074 x1_input_acceleration
   - 0.2243604029317559 x2_input_acceleration
   - 0.1178320559632025 x3_input_acceleration >= -0.0183107941310794
 CBF_Constraint_h_left: 0.1585926995148926 x1_input_acceleration
   - 0.2243604029317559 x2_input_acceleration
   - 0.1178320559632025 x3_input_acceleration >= -0.012343407038909
 CBF_Constraint_h_left: - 0.5414154067167969 x1_input_acceleration
   - 0.2243559921880425 x2_input_acceleration
   - 0.1178102406632147 x3_input_acceleration >= -0.07933637666653
 Relaxed_CLF_constraint: - 0.0408035477034913 x1_input_acceleration
   + 0.044410130565372 x2_input_acceleration
   - 0.0530339264458041 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2205519543876419 x1_input_acceleration
   - 0.2245581069364517 x2_input_acceleration
   - 0.1158201158101463 x3_input_acceleration >= -0.0181209518003418
 CBF_Constraint_h_left: 0.1594480456123581 x1_input_acceleration
   - 0.2245581069364517 x2_input_acceleration
   - 0.1158201158101463 x3_input_acceleration >= -0.0123160804666898
 CBF_Constraint_h_left: - 0.5405519543876419 x1_input_acceleration
   - 0.2245581069364517 x2_input_acceleration
   - 0.1158201158101463 x3_input_acceleration >= -0.0790092645023645
 Relaxed_CLF_constraint: - 0.04051081957053 x1_input_acceleration
   + 0.0439479286484733 x2_input_acceleration
   - 0.0526356257227611 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2202373197021453 x1_input_acceleration
   - 0.2262300749489552 x2_input_acceleration
   - 0.116242925318545 x3_input_acceleration >= -0.0182992353719019
 CBF_Constraint_h_left: 0.1597626802978547 x1_input_acceleration
   - 0.2262300749489552 x2_input_acceleration
   - 0.116242925318545 x3_input_acceleration >= -0.0125541446284943
 CBF_Constraint_h_left: - 0.5402373197021453 x1_input_acceleration
   - 0.2262300749489552 x2_input_acceleration
   - 0.116242925318545 x3_input_acceleration >= -0.0791372065242452
 Relaxed_CLF_constraint: - 0.040218080474852 x1_input_acceleration
   + 0.043485725369608 x2_input_acceleration
   - 0.052237334565553 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2196602845019067 x1_input_acceleration
   - 0.2271320814349238 x2_input_acceleration
   - 0.1154391886475702 x3_input_acceleration >= -0.0182914573200072
 CBF_Constraint_h_left: 0.1603397154980933 x1_input_acceleration
   - 0.2271320814349238 x2_input_acceleration
   - 0.1154391886475702 x3_input_acceleration >= -0.0126560032646449
 CBF_Constraint_h_left: - 0.5396602845019067 x1_input_acceleration
   - 0.2271320814349238 x2_input_acceleration
   - 0.1154391886475702 x3_input_acceleration >= -0.0790371028403123
 Relaxed_CLF_constraint: - 0.0399252713403089 x1_input_acceleration
   + 0.043023516453406 x2_input_acceleration
   - 0.0518391021133969 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2190681416302036 x1_input_acceleration
   - 0.2280691228677485 x2_input_acceleration
   - 0.1146565442326745 x3_input_acceleration >= -0.0182881246545886
 CBF_Constraint_h_left: 0.1609318583697964 x1_input_acceleration
   - 0.2280691228677485 x2_input_acceleration
   - 0.1146565442326745 x3_input_acceleration >= -0.0127651777448499
 CBF_Constraint_h_left: - 0.5390681416302037 x1_input_acceleration
   - 0.2280691228677485 x2_input_acceleration
   - 0.1146565442326745 x3_input_acceleration >= -0.0789390273154212
 Relaxed_CLF_constraint: - 0.0396325377645648 x1_input_acceleration
   + 0.0425613138293398 x2_input_acceleration
   - 0.0514408061625103 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2182092370179965 x1_input_acceleration
   - 0.2282569584362157 x2_input_acceleration
   - 0.1126501601532848 x3_input_acceleration >= -0.0181016421942723
 CBF_Constraint_h_left: 0.1617907629820035 x1_input_acceleration
   - 0.2282569584362157 x2_input_acceleration
   - 0.1126501601532848 x3_input_acceleration >= -0.012741887160853
 CBF_Constraint_h_left: - 0.5382092370179965 x1_input_acceleration
   - 0.2282569584362157 x2_input_acceleration
   - 0.1126501601532848 x3_input_acceleration >= -0.0786151201171517
 Relaxed_CLF_constraint: - 0.0393396993042864 x1_input_acceleration
   + 0.042099098945039 x2_input_acceleration
   - 0.0510426012672857 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2178740963300584 x1_input_acceleration
   - 0.2299117607745962 x2_input_acceleration
   - 0.1130301754880054 x3_input_acceleration >= -0.018276090041241
 CBF_Constraint_h_left: 0.1621259036699416 x1_input_acceleration
   - 0.2299117607745962 x2_input_acceleration
   - 0.1130301754880054 x3_input_acceleration >= -0.0129800117385299
 CBF_Constraint_h_left: - 0.5378740963300584 x1_input_acceleration
   - 0.2299117607745962 x2_input_acceleration
   - 0.1130301754880054 x3_input_acceleration >= -0.0787359454540504
 Relaxed_CLF_constraint: - 0.0390469592062891 x1_input_acceleration
   + 0.0416368954343876 x2_input_acceleration
   - 0.0506443110907129 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2173010047688913 x1_input_acceleration
   - 0.2308434401681883 x2_input_acceleration
   - 0.1122725841219947 x3_input_acceleration >= -0.018278388421921
 CBF_Constraint_h_left: 0.1626989952311087 x1_input_acceleration
   - 0.2308434401681883 x2_input_acceleration
   - 0.1122725841219947 x3_input_acceleration >= -0.0130911975158317
 CBF_Constraint_h_left: - 0.5373010047688913 x1_input_acceleration
   - 0.2308434401681883 x2_input_acceleration
   - 0.1122725841219947 x3_input_acceleration >= -0.0786465491849436
 Relaxed_CLF_constraint: - 0.0387542145426489 x1_input_acceleration
   + 0.0411746916209229 x2_input_acceleration
   - 0.050246024688866 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2164729528411192 x1_input_acceleration
   - 0.2310113787851138 x2_input_acceleration
   - 0.1102524907562366 x3_input_acceleration >= -0.0180956020394767
 CBF_Constraint_h_left: 0.163555657388373 x1_input_acceleration
   - 0.2310196042260928 x2_input_acceleration
   - 0.1102350861999622 x3_input_acceleration >= -0.0130680712075091
 CBF_Constraint_h_left: - 0.536444342611627 x1_input_acceleration
   - 0.2310196042260928 x2_input_acceleration
   - 0.1102350861999622 x3_input_acceleration >= -0.0783235911215786
 Relaxed_CLF_constraint: - 0.0384613052169464 x1_input_acceleration
   + 0.0407124712367155 x2_input_acceleration
   - 0.0498478792892511 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.215887566569784 x1_input_acceleration
   - 0.2319357693166597 x2_input_acceleration
   - 0.1094558245591589 x3_input_acceleration >= -0.0180955550044648
 CBF_Constraint_h_left: 0.164112433430216 x1_input_acceleration
   - 0.2319357693166597 x2_input_acceleration
   - 0.1094558245591589 x3_input_acceleration >= -0.0131769173562058
 CBF_Constraint_h_left: - 0.535887566569784 x1_input_acceleration
   - 0.2319357693166597 x2_input_acceleration
   - 0.1094558245591589 x3_input_acceleration >= -0.0782375656556302
 Relaxed_CLF_constraint: - 0.0381686120812788 x1_input_acceleration
   + 0.0402502759709426 x2_input_acceleration
   - 0.0494495461907122 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.215275356191564 x1_input_acceleration
   - 0.2328665854854297 x2_input_acceleration
   - 0.1086474536241864 x3_input_acceleration >= -0.0180936486995169
 CBF_Constraint_h_left: 0.164724643808436 x1_input_acceleration
   - 0.2328665854854297 x2_input_acceleration
   - 0.1086474536241864 x3_input_acceleration >= -0.0132913310231197
 CBF_Constraint_h_left: - 0.535275356191564 x1_input_acceleration
   - 0.2328665854854297 x2_input_acceleration
   - 0.1086474536241864 x3_input_acceleration >= -0.0781377056901671
 Relaxed_CLF_constraint: - 0.0378756199140194 x1_input_acceleration
   + 0.0397880865373754 x2_input_acceleration
   - 0.0490514403969931 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2149647147184305 x1_input_acceleration
   - 0.2345341093561755 x2_input_acceleration
   - 0.1090664029386775 x3_input_acceleration >= -0.0182778893188632
 CBF_Constraint_h_left: 0.1653099434846945 x1_input_acceleration
   - 0.2337909586451159 x2_input_acceleration
   - 0.1078507066038386 x3_input_acceleration >= -0.0134043411685156
 CBF_Constraint_h_left: - 0.5346900565153054 x1_input_acceleration
   - 0.2337909586451159 x2_input_acceleration
   - 0.1078507066038386 x3_input_acceleration >= -0.0780458609488725
 Relaxed_CLF_constraint: - 0.0375829700758901 x1_input_acceleration
   + 0.0393259 x2_input_acceleration
   - 0.0486530668868192 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2143764458120847 x1_input_acceleration
   - 0.2354393990114698 x2_input_acceleration
   - 0.1082615961878686 x3_input_acceleration >= -0.0182773860837622
 CBF_Constraint_h_left: 0.1656235541879153 x1_input_acceleration
   - 0.2354393990114698 x2_input_acceleration
   - 0.1082615961878686 x3_input_acceleration >= -0.0136458613794661
 CBF_Constraint_h_left: - 0.5343764458120848 x1_input_acceleration
   - 0.2354393990114698 x2_input_acceleration
   - 0.1082615961878686 x3_input_acceleration >= -0.0781776174136958
 Relaxed_CLF_constraint: - 0.0372895508784545 x1_input_acceleration
   + 0.0388650296889061 x2_input_acceleration
   - 0.0482542397901012 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.213792261249648 x1_input_acceleration
   - 0.2363469835470993 x2_input_acceleration
   - 0.1074561828766788 x3_input_acceleration >= -0.0182784647101192
 CBF_Constraint_h_left: 0.166207738750352 x1_input_acceleration
   - 0.2363469835470993 x2_input_acceleration
   - 0.1074561828766788 x3_input_acceleration >= -0.0137579350726861
 CBF_Constraint_h_left: - 0.533792261249648 x1_input_acceleration
   - 0.2363469835470993 x2_input_acceleration
   - 0.1074561828766788 x3_input_acceleration >= -0.0780852265100629
 Relaxed_CLF_constraint: - 0.036992213388496 x1_input_acceleration
   + 0.0384126083279213 x2_input_acceleration
   - 0.0478517119484407 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2128811542186431 x1_input_acceleration
   - 0.2364739422564239 x2_input_acceleration
   - 0.1054252511549326 x3_input_acceleration >= -0.0180881986922092
 CBF_Constraint_h_left: 0.1671188457813569 x1_input_acceleration
   - 0.2364739422564239 x2_input_acceleration
   - 0.1054252511549326 x3_input_acceleration >= -0.013740779390667
 CBF_Constraint_h_left: - 0.5331710712108306 x1_input_acceleration
   - 0.2372006420855255 x2_input_acceleration
   - 0.1066397553968759 x3_input_acceleration >= -0.0779768933032489
 Relaxed_CLF_constraint: - 0.0366813799505082 x1_input_acceleration
   + 0.0379885804770126 x2_input_acceleration
   - 0.0474370013022587 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2125189755830919 x1_input_acceleration
   - 0.2380339676534395 x2_input_acceleration
   - 0.1057751715076919 x3_input_acceleration >= -0.0182532179118018
 CBF_Constraint_h_left: 0.1677499605741347 x1_input_acceleration
   - 0.2373007113134127 x2_input_acceleration
   - 0.1045861780537124 x3_input_acceleration >= -0.0138474863755896
 CBF_Constraint_h_left: - 0.5322500394258654 x1_input_acceleration
   - 0.2373007113134127 x2_input_acceleration
   - 0.1045861780537124 x3_input_acceleration >= -0.0776350001746424
 Relaxed_CLF_constraint: - 0.0363658225541192 x1_input_acceleration
   + 0.0375751959485182 x2_input_acceleration
   - 0.0470174647516487 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2118926213982686 x1_input_acceleration
   - 0.238856772584471 x2_input_acceleration
   - 0.1049701423856016 x3_input_acceleration >= -0.0182424429012383
 CBF_Constraint_h_left: 0.1681073786017314 x1_input_acceleration
   - 0.238856772584471 x2_input_acceleration
   - 0.1049701423856016 x3_input_acceleration >= -0.0140828448355673
 CBF_Constraint_h_left: - 0.5318926213982687 x1_input_acceleration
   - 0.238856772584471 x2_input_acceleration
   - 0.1049701423856016 x3_input_acceleration >= -0.0777452623249613
 Relaxed_CLF_constraint: - 0.0360504996944957 x1_input_acceleration
   + 0.0371623629715279 x2_input_acceleration
   - 0.04659730721575 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.211270132695894 x1_input_acceleration
   - 0.2396941646697217 x2_input_acceleration
   - 0.1041241136678592 x3_input_acceleration >= -0.0182325481482934
 CBF_Constraint_h_left: 0.168729867304106 x1_input_acceleration
   - 0.2396941646697217 x2_input_acceleration
   - 0.1041241136678592 x3_input_acceleration >= -0.0141912229360735
 CBF_Constraint_h_left: - 0.531270132695894 x1_input_acceleration
   - 0.2396941646697217 x2_input_acceleration
   - 0.1041241136678592 x3_input_acceleration >= -0.0776357693796364
 Relaxed_CLF_constraint: - 0.0357354406460428 x1_input_acceleration
   + 0.0367495677887353 x2_input_acceleration
   - 0.0461769154612489 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2106461215423163 x1_input_acceleration
   - 0.2405195798177555 x2_input_acceleration
   - 0.1032946702348846 x3_input_acceleration >= -0.0182228114238659
 CBF_Constraint_h_left: 0.1696404575897638 x1_input_acceleration
   - 0.239766057898505 x2_input_acceleration
   - 0.1020638343202728 x3_input_acceleration >= -0.0141706684119024
 CBF_Constraint_h_left: - 0.5303595424102363 x1_input_acceleration
   - 0.239766057898505 x2_input_acceleration
   - 0.1020638343202728 x3_input_acceleration >= -0.0772965082554851
 Relaxed_CLF_constraint: - 0.0354205740463047 x1_input_acceleration
   + 0.0363368006100736 x2_input_acceleration
   - 0.0457563524895736 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2097438785708362 x1_input_acceleration
   - 0.2405969220961317 x2_input_acceleration
   - 0.1012383491420256 x3_input_acceleration >= -0.0180321442142681
 CBF_Constraint_h_left: 0.1702561214291639 x1_input_acceleration
   - 0.2405969220961317 x2_input_acceleration
   - 0.1012383491420256 x3_input_acceleration >= -0.0142808072858092
 CBF_Constraint_h_left: - 0.5297438785708362 x1_input_acceleration
   - 0.2405969220961317 x2_input_acceleration
   - 0.1012383491420256 x3_input_acceleration >= -0.0771911647856019
 Relaxed_CLF_constraint: - 0.0351058279878107 x1_input_acceleration
   + 0.0359240507202752 x2_input_acceleration
   - 0.0453356824826949 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2093992265782447 x1_input_acceleration
   - 0.242163164221076 x2_input_acceleration
   - 0.1016036152484923 x3_input_acceleration >= -0.0182035822071736
 CBF_Constraint_h_left: 0.1706007734217553 x1_input_acceleration
   - 0.242163164221076 x2_input_acceleration
   - 0.1016036152484923 x3_input_acceleration >= -0.0145177291573071
 CBF_Constraint_h_left: - 0.5293992265782448 x1_input_acceleration
   - 0.242163164221076 x2_input_acceleration
   - 0.1016036152484923 x3_input_acceleration >= -0.0773074584596928
 Relaxed_CLF_constraint: - 0.0347911660997914 x1_input_acceleration
   + 0.0355113129204373 x2_input_acceleration
   - 0.044914937716867 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2087428343152539 x1_input_acceleration
   - 0.2429730069018166 x2_input_acceleration
   - 0.1007582118110212 x3_input_acceleration >= -0.0181904175520576
 CBF_Constraint_h_left: 0.1712571656847461 x1_input_acceleration
   - 0.2429730069018166 x2_input_acceleration
   - 0.1007582118110212 x3_input_acceleration >= -0.0146292790321593
 CBF_Constraint_h_left: - 0.5287428343152539 x1_input_acceleration
   - 0.2429730069018166 x2_input_acceleration
   - 0.1007582118110212 x3_input_acceleration >= -0.0771892710424982
 Relaxed_CLF_constraint: - 0.0344765598273048 x1_input_acceleration
   + 0.0350985833716014 x2_input_acceleration
   - 0.0444941433467642 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2078566782235393 x1_input_acceleration
   - 0.2430753321959893 x2_input_acceleration
   - 0.0986951410612077 x3_input_acceleration >= -0.0180076866683515
 CBF_Constraint_h_left: 0.1721433217764607 x1_input_acceleration
   - 0.2430753321959893 x2_input_acceleration
   - 0.0986951410612077 x3_input_acceleration >= -0.0146149178058791
 CBF_Constraint_h_left: - 0.5278566782235393 x1_input_acceleration
   - 0.2430753321959893 x2_input_acceleration
   - 0.0986951410612077 x3_input_acceleration >= -0.0768647551841178
 Relaxed_CLF_constraint: - 0.0341619893955832 x1_input_acceleration
   + 0.0346858589493075 x2_input_acceleration
   - 0.0440733171622261 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2075031810065071 x1_input_acceleration
   - 0.2446533453848037 x2_input_acceleration
   - 0.0990728528737862 x3_input_acceleration >= -0.0181820649280865
 CBF_Constraint_h_left: 0.172496818993493 x1_input_acceleration
   - 0.2446533453848037 x2_input_acceleration
   - 0.0990728528737862 x3_input_acceleration >= -0.0148564605368502
 CBF_Constraint_h_left: - 0.5275031810065071 x1_input_acceleration
   - 0.2446533453848037 x2_input_acceleration
   - 0.0990728528737862 x3_input_acceleration >= -0.0769825738891277
 Relaxed_CLF_constraint: - 0.0338474348483464 x1_input_acceleration
   + 0.0342731364773674 x2_input_acceleration
   - 0.0436524771571079 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2065989901444708 x1_input_acceleration
   - 0.2447164387237077 x2_input_acceleration
   - 0.0970248462704232 x3_input_acceleration >= -0.0179957747260321
 CBF_Constraint_h_left: 0.1734010098555293 x1_input_acceleration
   - 0.2447164387237077 x2_input_acceleration
   - 0.0970248462704232 x3_input_acceleration >= -0.0148419665985827
 CBF_Constraint_h_left: - 0.5265989901444708 x1_input_acceleration
   - 0.2447164387237077 x2_input_acceleration
   - 0.0970248462704232 x3_input_acceleration >= -0.0766516131491475
 Relaxed_CLF_constraint: - 0.0335329069374698 x1_input_acceleration
   + 0.0338604181045767 x2_input_acceleration
   - 0.0432316132559086 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.206234126047637 x1_input_acceleration
   - 0.2462768009554597 x2_input_acceleration
   - 0.0973596958074208 x3_input_acceleration >= -0.0181659219508003
 CBF_Constraint_h_left: 0.1737673044638376 x1_input_acceleration
   - 0.2463000467669221 x2_input_acceleration
   - 0.0973852065953846 x3_input_acceleration >= -0.0150856669004104
 CBF_Constraint_h_left: - 0.5262326955361624 x1_input_acceleration
   - 0.2463000467669221 x2_input_acceleration
   - 0.0973852065953846 x3_input_acceleration >= -0.0767671103380673
 Relaxed_CLF_constraint: - 0.0332183909962307 x1_input_acceleration
   + 0.0334476996237411 x2_input_acceleration
   - 0.0428107401804743 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2056051142954943 x1_input_acceleration
   - 0.2471022313571583 x2_input_acceleration
   - 0.0965179288551039 x3_input_acceleration >= -0.0181621715891572
 CBF_Constraint_h_left: 0.1743948857045057 x1_input_acceleration
   - 0.2471022313571583 x2_input_acceleration
   - 0.0965179288551039 x3_input_acceleration >= -0.0151971998730133
 CBF_Constraint_h_left: - 0.5256051142954943 x1_input_acceleration
   - 0.2471022313571583 x2_input_acceleration
   - 0.0965179288551039 x3_input_acceleration >= -0.0766589898764363
 Relaxed_CLF_constraint: - 0.0329038851201593 x1_input_acceleration
   + 0.0330349844228918 x2_input_acceleration
   - 0.0423898567043158 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2047204816646353 x1_input_acceleration
   - 0.2471947770253056 x2_input_acceleration
   - 0.0944870883577874 x3_input_acceleration >= -0.0179858858169808
 CBF_Constraint_h_left: 0.1752838098697885 x1_input_acceleration
   - 0.2472090821400517 x2_input_acceleration
   - 0.0945110494249871 x3_input_acceleration >= -0.0151922706896018
 CBF_Constraint_h_left: - 0.5247161901302115 x1_input_acceleration
   - 0.2472090821400517 x2_input_acceleration
   - 0.0945110494249871 x3_input_acceleration >= -0.0763429372351759
 Relaxed_CLF_constraint: - 0.0325893611631283 x1_input_acceleration
   + 0.032622262232739 x2_input_acceleration
   - 0.0419689927872921 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2043455394833345 x1_input_acceleration
   - 0.2487648006165859 x2_input_acceleration
   - 0.0948564441202966 x3_input_acceleration >= -0.0181596926309229
 CBF_Constraint_h_left: 0.1756544605166655 x1_input_acceleration
   - 0.2487648006165859 x2_input_acceleration
   - 0.0948564441202966 x3_input_acceleration >= -0.0154340401290894
 CBF_Constraint_h_left: - 0.5243455394833345 x1_input_acceleration
   - 0.2487648006165859 x2_input_acceleration
   - 0.0948564441202966 x3_input_acceleration >= -0.0764549789482565
 Relaxed_CLF_constraint: - 0.0322748129698167 x1_input_acceleration
   + 0.0322095410483659 x2_input_acceleration
   - 0.0415481470822735 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2037127774955662 x1_input_acceleration
   - 0.2495976125846864 x2_input_acceleration
   - 0.0940168847705029 x3_input_acceleration >= -0.0181592596362208
 CBF_Constraint_h_left: 0.1762872225044338 x1_input_acceleration
   - 0.2495976125846864 x2_input_acceleration
   - 0.0940168847705029 x3_input_acceleration >= -0.0155538319120632
 CBF_Constraint_h_left: - 0.5237127774955662 x1_input_acceleration
   - 0.2495976125846864 x2_input_acceleration
   - 0.0940168847705029 x3_input_acceleration >= -0.0763533040355114
 Relaxed_CLF_constraint: - 0.0319603393245654 x1_input_acceleration
   + 0.0317968305522918 x2_input_acceleration
   - 0.0411272349191259 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2028344318389833 x1_input_acceleration
   - 0.249676546679918 x2_input_acceleration
   - 0.0919861972265632 x3_input_acceleration >= -0.0179854112954142
 CBF_Constraint_h_left: 0.1771655681610167 x1_input_acceleration
   - 0.249676546679918 x2_input_acceleration
   - 0.0919861972265632 x3_input_acceleration >= -0.0155468692460073
 CBF_Constraint_h_left: - 0.5228344318389833 x1_input_acceleration
   - 0.249676546679918 x2_input_acceleration
   - 0.0919861972265632 x3_input_acceleration >= -0.0760389203896515
 Relaxed_CLF_constraint: - 0.0316458594131441 x1_input_acceleration
   + 0.0313841192190925 x2_input_acceleration
   - 0.0407063282695011 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2024591382101542 x1_input_acceleration
   - 0.2512395188176562 x2_input_acceleration
   - 0.0923406590670546 x3_input_acceleration >= -0.0181594489443659
 CBF_Constraint_h_left: 0.1775408617898458 x1_input_acceleration
   - 0.2512395188176562 x2_input_acceleration
   - 0.0923406590670546 x3_input_acceleration >= -0.0157922126844366
 CBF_Constraint_h_left: - 0.5224591382101542 x1_input_acceleration
   - 0.2512395188176562 x2_input_acceleration
   - 0.0923406590670546 x3_input_acceleration >= -0.0761529110579906
 Relaxed_CLF_constraint: - 0.0313313739146475 x1_input_acceleration
   + 0.0309714070874487 x2_input_acceleration
   - 0.0402854265794257 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2016317342517977 x1_input_acceleration
   - 0.2520116682238249 x2_input_acceleration
   - 0.0910493799486851 x3_input_acceleration >= -0.0181138066918457
 CBF_Constraint_h_left: 0.1784407449962492 x1_input_acceleration
   - 0.251338135737863 x2_input_acceleration
   - 0.0902574726381993 x3_input_acceleration >= -0.015789592329509
 CBF_Constraint_h_left: - 0.5215592550037509 x1_input_acceleration
   - 0.251338135737863 x2_input_acceleration
   - 0.0902574726381993 x3_input_acceleration >= -0.0758353315808218
 Relaxed_CLF_constraint: - 0.0310168540643755 x1_input_acceleration
   + 0.0305586891081497 x2_input_acceleration
   - 0.039864556135689 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2009245763261833 x1_input_acceleration
   - 0.252157716809347 x2_input_acceleration
   - 0.0894375911085918 x3_input_acceleration >= -0.0179883205554167
 CBF_Constraint_h_left: 0.1790754236738167 x1_input_acceleration
   - 0.252157716809347 x2_input_acceleration
   - 0.0894375911085918 x3_input_acceleration >= -0.0159126510534419
 CBF_Constraint_h_left: - 0.5209245763261833 x1_input_acceleration
   - 0.252157716809347 x2_input_acceleration
   - 0.0894375911085918 x3_input_acceleration >= -0.075736252767606
 Relaxed_CLF_constraint: - 0.0307023757485409 x1_input_acceleration
   + 0.0301459779448138 x2_input_acceleration
   - 0.0394436481162466 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.2005845721642147 x1_input_acceleration
   - 0.2536987538939692 x2_input_acceleration
   - 0.0898160108644216 x3_input_acceleration >= -0.0181660360313129
 CBF_Constraint_h_left: 0.1794154278357853 x1_input_acceleration
   - 0.2536987538939692 x2_input_acceleration
   - 0.0898160108644216 x3_input_acceleration >= -0.0161549673201121
 CBF_Constraint_h_left: - 0.5205845721642146 x1_input_acceleration
   - 0.2536987538939692 x2_input_acceleration
   - 0.0898160108644216 x3_input_acceleration >= -0.0758595675775872
 Relaxed_CLF_constraint: - 0.0303878920086211 x1_input_acceleration
   + 0.0297332660850893 x2_input_acceleration
   - 0.0390227448502406 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1999455833302194 x1_input_acceleration
   - 0.2545576766150599 x2_input_acceleration
   - 0.0889928079467904 x3_input_acceleration >= -0.0181743917207934
 CBF_Constraint_h_left: 0.1800444030894583 x1_input_acceleration
   - 0.2545241788046961 x2_input_acceleration
   - 0.0889742112976205 x3_input_acceleration >= -0.0162787387390195
 CBF_Constraint_h_left: - 0.5199555969105417 x1_input_acceleration
   - 0.2545241788046961 x2_input_acceleration
   - 0.0889742112976205 x3_input_acceleration >= -0.0757631976577091
 Relaxed_CLF_constraint: - 0.0300734043817846 x1_input_acceleration
   + 0.0293205536297259 x2_input_acceleration
   - 0.03860184506684 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1993204194137104 x1_input_acceleration
   - 0.2553730890500843 x2_input_acceleration
   - 0.0880997518628455 x3_input_acceleration >= -0.0181764026211337
 CBF_Constraint_h_left: 0.1806833952835552 x1_input_acceleration
   - 0.2553831026304065 x2_input_acceleration
   - 0.0881510118573523 x3_input_acceleration >= -0.0164094048329503
 CBF_Constraint_h_left: - 0.5193166047164448 x1_input_acceleration
   - 0.2553831026304065 x2_input_acceleration
   - 0.0881510118573523 x3_input_acceleration >= -0.075670216483706
 Relaxed_CLF_constraint: - 0.0297589150748973 x1_input_acceleration
   + 0.0289078406220525 x2_input_acceleration
   - 0.038180947022121 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1987043116105269 x1_input_acceleration
   - 0.2562005415871116 x2_input_acceleration
   - 0.0872903836042071 x3_input_acceleration >= -0.0181854330079781
 CBF_Constraint_h_left: 0.1812956883894731 x1_input_acceleration
   - 0.2562005415871116 x2_input_acceleration
   - 0.0872903836042071 x3_input_acceleration >= -0.016531613801978
 CBF_Constraint_h_left: - 0.518704311610527 x1_input_acceleration
   - 0.2562005415871116 x2_input_acceleration
   - 0.0872903836042071 x3_input_acceleration >= -0.0755781228656624
 Relaxed_CLF_constraint: - 0.0294444238716698 x1_input_acceleration
   + 0.0284951276325001 x2_input_acceleration
   - 0.0377600504294228 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1977638162178645 x1_input_acceleration
   - 0.256286533011709 x2_input_acceleration
   - 0.0852168053849504 x3_input_acceleration >= -0.0180138044820579
 CBF_Constraint_h_left: 0.1822361837821355 x1_input_acceleration
   - 0.256286533011709 x2_input_acceleration
   - 0.0852168053849504 x3_input_acceleration >= -0.0165386794006636
 CBF_Constraint_h_left: - 0.5177638162178646 x1_input_acceleration
   - 0.256286533011709 x2_input_acceleration
   - 0.0852168053849504 x3_input_acceleration >= -0.0752560150769163
 Relaxed_CLF_constraint: - 0.0291298635074919 x1_input_acceleration
   + 0.0280824043086596 x2_input_acceleration
   - 0.0373392157189034 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1974343306003795 x1_input_acceleration
   - 0.2578431824800143 x2_input_acceleration
   - 0.085569093327889 x3_input_acceleration >= -0.0181963728459997
 CBF_Constraint_h_left: 0.1825656693996205 x1_input_acceleration
   - 0.2578431824800143 x2_input_acceleration
   - 0.085569093327889 x3_input_acceleration >= -0.0167838500319276
 CBF_Constraint_h_left: - 0.5174343306003795 x1_input_acceleration
   - 0.2578431824800143 x2_input_acceleration
   - 0.085569093327889 x3_input_acceleration >= -0.0753858657420604
 Relaxed_CLF_constraint: - 0.0288153938310736 x1_input_acceleration
   + 0.0276696944655592 x2_input_acceleration
   - 0.0369183 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1967948833485927 x1_input_acceleration
   - 0.2586748045516634 x2_input_acceleration
   - 0.0847667364287579 x3_input_acceleration >= -0.0182065700567025
 CBF_Constraint_h_left: 0.1831946262339268 x1_input_acceleration
   - 0.2586686056686067 x2_input_acceleration
   - 0.0847272781539166 x3_input_acceleration >= -0.0169121075757251
 CBF_Constraint_h_left: - 0.5168053737660732 x1_input_acceleration
   - 0.2586686056686067 x2_input_acceleration
   - 0.0847272781539166 x3_input_acceleration >= -0.0752939883938507
 Relaxed_CLF_constraint: - 0.0285009154139204 x1_input_acceleration
   + 0.027256982871263 x2_input_acceleration
   - 0.0364973923766714 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1959060429469829 x1_input_acceleration
   - 0.2587480171574699 x2_input_acceleration
   - 0.0826921836724296 x3_input_acceleration >= -0.0180419278216456
 CBF_Constraint_h_left: 0.1838340808017964 x1_input_acceleration
   - 0.2595002277745353 x2_input_acceleration
   - 0.0839249269356742 x3_input_acceleration >= -0.0170446827101088
 CBF_Constraint_h_left: - 0.5161659191982036 x1_input_acceleration
   - 0.2595002277745353 x2_input_acceleration
   - 0.0839249269356742 x3_input_acceleration >= -0.0752027544294801
 Relaxed_CLF_constraint: - 0.0281864333387259 x1_input_acceleration
   + 0.026844271259827 x2_input_acceleration
   - 0.0360764876301296 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1952770660136273 x1_input_acceleration
   - 0.2595734422118914 x2_input_acceleration
   - 0.0818503855220122 x3_input_acceleration >= -0.0180527475056807
 CBF_Constraint_h_left: 0.1847229339863727 x1_input_acceleration
   - 0.2595734422118914 x2_input_acceleration
   - 0.0818503855220122 x3_input_acceleration >= -0.0170501049630915
 CBF_Constraint_h_left: - 0.5152770660136272 x1_input_acceleration
   - 0.2595734422118914 x2_input_acceleration
   - 0.0818503855220122 x3_input_acceleration >= -0.074897078067861
 Relaxed_CLF_constraint: - 0.0278719448720481 x1_input_acceleration
   + 0.0264315587326163 x2_input_acceleration
   - 0.0356555885549209 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1949108284190068 x1_input_acceleration
   - 0.2611327193719888 x2_input_acceleration
   - 0.0822062812808359 x3_input_acceleration >= -0.0182346002109043
 CBF_Constraint_h_left: 0.1853523856923213 x1_input_acceleration
   - 0.2604047082407022 x2_input_acceleration
   - 0.0810171686175668 x3_input_acceleration >= -0.0171824751416419
 CBF_Constraint_h_left: - 0.5146476143076787 x1_input_acceleration
   - 0.2604047082407022 x2_input_acceleration
   - 0.0810171686175668 x3_input_acceleration >= -0.0748091401493294
 Relaxed_CLF_constraint: - 0.0275574574376529 x1_input_acceleration
   + 0.0260188463458049 x2_input_acceleration
   - 0.0352346885682744 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1940172022976152 x1_input_acceleration
   - 0.2612273920677723 x2_input_acceleration
   - 0.0801990971225675 x3_input_acceleration >= -0.0180785800832996
 CBF_Constraint_h_left: 0.1857181530795821 x1_input_acceleration
   - 0.2619581450127186 x2_input_acceleration
   - 0.081364487137216 x3_input_acceleration >= -0.0174333704722244
 CBF_Constraint_h_left: - 0.5142818469204179 x1_input_acceleration
   - 0.2619581450127186 x2_input_acceleration
   - 0.081364487137216 x3_input_acceleration >= -0.0749320168943707
 Relaxed_CLF_constraint: - 0.0272429666883584 x1_input_acceleration
   + 0.0256061335254399 x2_input_acceleration
   - 0.0348137914964644 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1933867776851291 x1_input_acceleration
   - 0.2620540107216869 x2_input_acceleration
   - 0.0793241734017776 x3_input_acceleration >= -0.0180907687011587
 CBF_Constraint_h_left: 0.1866117918033963 x1_input_acceleration
   - 0.2620528186287914 x2_input_acceleration
   - 0.0793573135842727 x3_input_acceleration >= -0.0174483059526702
 CBF_Constraint_h_left: - 0.5133882081966037 x1_input_acceleration
   - 0.2620528186287914 x2_input_acceleration
   - 0.0793573135842727 x3_input_acceleration >= -0.0746341788214815
 Relaxed_CLF_constraint: - 0.0269284696378527 x1_input_acceleration
   + 0.0251934202449304 x2_input_acceleration
   - 0.034392899727317 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1927620852578956 x1_input_acceleration
   - 0.2628848018933787 x2_input_acceleration
   - 0.0784797594360901 x3_input_acceleration >= -0.0181061783051599
 CBF_Constraint_h_left: 0.1872379147421044 x1_input_acceleration
   - 0.2628848018933787 x2_input_acceleration
   - 0.0784797594360901 x3_input_acceleration >= -0.0175813821061598
 CBF_Constraint_h_left: - 0.5127620852578956 x1_input_acceleration
   - 0.2628848018933787 x2_input_acceleration
   - 0.0784797594360901 x3_input_acceleration >= -0.0745481119464232
 Relaxed_CLF_constraint: - 0.0266139776570241 x1_input_acceleration
   + 0.0247807068680994 x2_input_acceleration
   - 0.0339720040466583 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1923972697953771 x1_input_acceleration
   - 0.2644531401827752 x2_input_acceleration
   - 0.078864511299821 x3_input_acceleration >= -0.0182929459799513
 CBF_Constraint_h_left: 0.1876027302046229 x1_input_acceleration
   - 0.2644531401827752 x2_input_acceleration
   - 0.078864511299821 x3_input_acceleration >= -0.0178374647188297
 CBF_Constraint_h_left: - 0.5123972697953771 x1_input_acceleration
   - 0.2644531401827752 x2_input_acceleration
   - 0.078864511299821 x3_input_acceleration >= -0.0746765091472117
 Relaxed_CLF_constraint: - 0.0262994860329425 x1_input_acceleration
   + 0.0243679938696414 x2_input_acceleration
   - 0.0335511077884298 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1917768725328969 x1_input_acceleration
   - 0.2652662872171159 x2_input_acceleration
   - 0.0780209280594484 x3_input_acceleration >= -0.0183079592969275
 CBF_Constraint_h_left: 0.1882231274671031 x1_input_acceleration
   - 0.2652662872171159 x2_input_acceleration
   - 0.0780209280594484 x3_input_acceleration >= -0.0179703535156771
 CBF_Constraint_h_left: - 0.5117768725328968 x1_input_acceleration
   - 0.2652662872171159 x2_input_acceleration
   - 0.0780209280594484 x3_input_acceleration >= -0.074592258902191
 Relaxed_CLF_constraint: - 0.0259849958672786 x1_input_acceleration
   + 0.0239552810740592 x2_input_acceleration
   - 0.0331302102379151 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1911168941188427 x1_input_acceleration
   - 0.2660939784000256 x2_input_acceleration
   - 0.077167692217452 x3_input_acceleration >= -0.0183216313201384
 CBF_Constraint_h_left: 0.1888831058811573 x1_input_acceleration
   - 0.2660939784000256 x2_input_acceleration
   - 0.077167692217452 x3_input_acceleration >= -0.0181094214375583
 CBF_Constraint_h_left: - 0.5111478885341258 x1_input_acceleration
   - 0.2660932631442883 x2_input_acceleration
   - 0.0771839046808309 x3_input_acceleration >= -0.0745087859463625
 Relaxed_CLF_constraint: - 0.0256705038678931 x1_input_acceleration
   + 0.0235425679708551 x2_input_acceleration
   - 0.0327093143628153 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1902411777309171 x1_input_acceleration
   - 0.2661830726827075 x2_input_acceleration
   - 0.0751178949479385 x3_input_acceleration >= -0.0181719580071659
 CBF_Constraint_h_left: 0.1895122974582918 x1_input_acceleration
   - 0.2669194284642626 x2_input_acceleration
   - 0.0763260810975357 x3_input_acceleration >= -0.0182466407088291
 CBF_Constraint_h_left: - 0.5104877025417082 x1_input_acceleration
   - 0.2669194284642626 x2_input_acceleration
   - 0.0763260810975357 x3_input_acceleration >= -0.0744173365984269
 Relaxed_CLF_constraint: - 0.0253559080793258 x1_input_acceleration
   + 0.0231298429387365 x2_input_acceleration
   - 0.0322885088028572 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1896146407407549 x1_input_acceleration
   - 0.2670058669251767 x2_input_acceleration
   - 0.0742860607259398 x3_input_acceleration >= -0.0181910659434722
 CBF_Constraint_h_left: 0.1901331124025556 x1_input_acceleration
   - 0.2677514018220273 x2_input_acceleration
   - 0.0754762462728148 x3_input_acceleration >= -0.0183845193402445
 CBF_Constraint_h_left: - 0.5098668875974444 x1_input_acceleration
   - 0.2677514018220273 x2_input_acceleration
   - 0.0754762462728148 x3_input_acceleration >= -0.0743379299993501
 Relaxed_CLF_constraint: - 0.0250414474913492 x1_input_acceleration
   + 0.0227171345153169 x2_input_acceleration
   - 0.0318675849016967 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1892431829171781 x1_input_acceleration
   - 0.268564901189839 x2_input_acceleration
   - 0.0746625569104283 x3_input_acceleration >= -0.0183786464590339
 CBF_Constraint_h_left: 0.1907568170828219 x1_input_acceleration
   - 0.268564901189839 x2_input_acceleration
   - 0.0746625569104283 x3_input_acceleration >= -0.01852244170477
 CBF_Constraint_h_left: - 0.5092431829171781 x1_input_acceleration
   - 0.268564901189839 x2_input_acceleration
   - 0.0746625569104283 x3_input_acceleration >= -0.0742575557257824
 Relaxed_CLF_constraint: - 0.0247269725468459 x1_input_acceleration
   + 0.0223044243669335 x2_input_acceleration
   - 0.0314466735243365 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1883495814764061 x1_input_acceleration
   - 0.2686612391091214 x2_input_acceleration
   - 0.0725924082378086 x3_input_acceleration >= -0.0182310209939376
 CBF_Constraint_h_left: 0.1916504185235939 x1_input_acceleration
   - 0.2686612391091214 x2_input_acceleration
   - 0.0725924082378086 x3_input_acceleration >= -0.0185446005134205
 CBF_Constraint_h_left: - 0.5083495814764061 x1_input_acceleration
   - 0.2686612391091214 x2_input_acceleration
   - 0.0725924082378086 x3_input_acceleration >= -0.0739669540301626
 Relaxed_CLF_constraint: - 0.0244124941378613 x1_input_acceleration
   + 0.021891713399846 x2_input_acceleration
   - 0.0310257654477922 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1877225211099655 x1_input_acceleration
   - 0.2694904750371181 x2_input_acceleration
   - 0.0717610906762678 x3_input_acceleration >= -0.0182536288006651
 CBF_Constraint_h_left: 0.1922774788900345 x1_input_acceleration
   - 0.2694904750371181 x2_input_acceleration
   - 0.0717610906762678 x3_input_acceleration >= -0.0186863497897717
 CBF_Constraint_h_left: - 0.5077225211099654 x1_input_acceleration
   - 0.2694904750371181 x2_input_acceleration
   - 0.0717610906762678 x3_input_acceleration >= -0.0738892321782596
 Relaxed_CLF_constraint: - 0.0240980102803246 x1_input_acceleration
   + 0.0214790027844804 x2_input_acceleration
   - 0.0306048614582816 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1873591475155176 x1_input_acceleration
   - 0.2710304369507819 x2_input_acceleration
   - 0.0721118558184212 x3_input_acceleration >= -0.0184402669152625
 CBF_Constraint_h_left: 0.1926408524844824 x1_input_acceleration
   - 0.2710304369507819 x2_input_acceleration
   - 0.0721118558184212 x3_input_acceleration >= -0.0189420288873142
 CBF_Constraint_h_left: - 0.5073591475155176 x1_input_acceleration
   - 0.2710304369507819 x2_input_acceleration
   - 0.0721118558184212 x3_input_acceleration >= -0.0740177305177453
 Relaxed_CLF_constraint: - 0.0237835243345411 x1_input_acceleration
   + 0.0210662920684322 x2_input_acceleration
   - 0.0301839591630252 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1864640816880139 x1_input_acceleration
   - 0.2711364245682504 x2_input_acceleration
   - 0.0700621144817891 x3_input_acceleration >= -0.018298128593267
 CBF_Constraint_h_left: 0.1935359183119861 x1_input_acceleration
   - 0.2711364245682504 x2_input_acceleration
   - 0.0700621144817891 x3_input_acceleration >= -0.0189699530725444
 CBF_Constraint_h_left: - 0.5064640816880139 x1_input_acceleration
   - 0.2711364245682504 x2_input_acceleration
   - 0.0700621144817891 x3_input_acceleration >= -0.0737323816633493
 Relaxed_CLF_constraint: - 0.0234690289879279 x1_input_acceleration
   + 0.0206535842284785 x2_input_acceleration
   - 0.0297630623602183 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1858388314459182 x1_input_acceleration
   - 0.2719648648678012 x2_input_acceleration
   - 0.0692480238681061 x3_input_acceleration >= -0.0183240619513459
 CBF_Constraint_h_left: 0.1941611685540818 x1_input_acceleration
   - 0.2719648648678012 x2_input_acceleration
   - 0.0692480238681061 x3_input_acceleration >= -0.0191146839766214
 CBF_Constraint_h_left: - 0.5058388314459181 x1_input_acceleration
   - 0.2719648648678012 x2_input_acceleration
   - 0.0692480238681061 x3_input_acceleration >= -0.0736582749826928
 Relaxed_CLF_constraint: - 0.0231544965182472 x1_input_acceleration
   + 0.020240973404112 x2_input_acceleration
   - 0.0293421291711441 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.185460337292026 x1_input_acceleration
   - 0.2735089759874276 x2_input_acceleration
   - 0.0695932657235908 x3_input_acceleration >= -0.0185114798220594
 CBF_Constraint_h_left: 0.194539662707974 x1_input_acceleration
   - 0.2735089759874276 x2_input_acceleration
   - 0.0695932657235908 x3_input_acceleration >= -0.0193740157365745
 CBF_Constraint_h_left: - 0.5051999842036472 x1_input_acceleration
   - 0.2727714281129769 x2_input_acceleration
   - 0.0683924705499458 x3_input_acceleration >= -0.0735772015155209
 Relaxed_CLF_constraint: - 0.0228362649900072 x1_input_acceleration
   + 0.0198367977118525 x2_input_acceleration
   - 0.0289184056279086 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.184820311743813 x1_input_acceleration
   - 0.2742704218449858 x2_input_acceleration
   - 0.06876041165298 x3_input_acceleration >= -0.0185277015356985
 CBF_Constraint_h_left: 0.195179688256187 x1_input_acceleration
   - 0.2742704218449858 x2_input_acceleration
   - 0.06876041165298 x3_input_acceleration >= -0.019511842304374
 CBF_Constraint_h_left: - 0.504820311743813 x1_input_acceleration
   - 0.2742704218449858 x2_input_acceleration
   - 0.06876041165298 x3_input_acceleration >= -0.0736989514147086
 Relaxed_CLF_constraint: - 0.0225043312869456 x1_input_acceleration
   + 0.019463882991539 x2_input_acceleration
   - 0.0284843356177497 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1841360920615023 x1_input_acceleration
   - 0.2749539072214087 x2_input_acceleration
   - 0.0678603452850018 x3_input_acceleration >= -0.0185276944895502
 CBF_Constraint_h_left: 0.1958639079384977 x1_input_acceleration
   - 0.2749539072214087 x2_input_acceleration
   - 0.0678603452850018 x3_input_acceleration >= -0.0196418369978647
 CBF_Constraint_h_left: - 0.504109389180643 x1_input_acceleration
   - 0.2749626094995459 x2_input_acceleration
   - 0.0678438944030438 x3_input_acceleration >= -0.073583376722663
 Relaxed_CLF_constraint: - 0.0221631751201067 x1_input_acceleration
   + 0.0191120075137158 x2_input_acceleration
   - 0.0280433027351217 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1831648041147371 x1_input_acceleration
   - 0.274896685869001 x2_input_acceleration
   - 0.0657520210097895 x3_input_acceleration >= -0.0183602154087555
 CBF_Constraint_h_left: 0.1965586303335051 x1_input_acceleration
   - 0.2756293461625801 x2_input_acceleration
   - 0.0669689094375239 x3_input_acceleration >= -0.0197729166139615
 CBF_Constraint_h_left: - 0.503441369666495 x1_input_acceleration
   - 0.2756293461625801 x2_input_acceleration
   - 0.0669689094375239 x3_input_acceleration >= -0.0734773959972348
 Relaxed_CLF_constraint: - 0.021815813922603 x1_input_acceleration
   + 0.0187742880431301 x2_input_acceleration
   - 0.0275975848113828 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.182469361563957 x1_input_acceleration
   - 0.2755427067342753 x2_input_acceleration
   - 0.0648565469812056 x3_input_acceleration >= -0.0183563057075836
 CBF_Constraint_h_left: 0.197530638436043 x1_input_acceleration
   - 0.2755427067342753 x2_input_acceleration
   - 0.0648565469812056 x3_input_acceleration >= -0.0197871270104317
 CBF_Constraint_h_left: - 0.5024693615639571 x1_input_acceleration
   - 0.2755427067342753 x2_input_acceleration
   - 0.0648565469812056 x3_input_acceleration >= -0.0731514035578167
 Relaxed_CLF_constraint: - 0.0214642779499473 x1_input_acceleration
   + 0.0184460920063975 x2_input_acceleration
   - 0.0271487153088401 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.18200710815536 x1_input_acceleration
   - 0.2769277763709628 x2_input_acceleration
   - 0.0651532054603352 x3_input_acceleration >= -0.0185151302316499
 CBF_Constraint_h_left: 0.19799289184464 x1_input_acceleration
   - 0.2769277763709628 x2_input_acceleration
   - 0.0651532054603352 x3_input_acceleration >= -0.0200337796821315
 CBF_Constraint_h_left: - 0.50200710815536 x1_input_acceleration
   - 0.2769277763709628 x2_input_acceleration
   - 0.0651532054603352 x3_input_acceleration >= -0.0732362675365075
 Relaxed_CLF_constraint: - 0.0211101272588324 x1_input_acceleration
   + 0.0181238212413984 x2_input_acceleration
   - 0.0266978986591226 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1813007332503817 x1_input_acceleration
   - 0.2775741515039662 x2_input_acceleration
   - 0.0642511381969815 x3_input_acceleration >= -0.0185113935549701
 CBF_Constraint_h_left: 0.1986992667496184 x1_input_acceleration
   - 0.2775741515039662 x2_input_acceleration
   - 0.0642511381969815 x3_input_acceleration >= -0.0201642542373976
 CBF_Constraint_h_left: - 0.5013007332503816 x1_input_acceleration
   - 0.2775741515039662 x2_input_acceleration
   - 0.0642511381969815 x3_input_acceleration >= -0.0731195108750312
 Relaxed_CLF_constraint: - 0.0207538403648149 x1_input_acceleration
   + 0.0178064749300847 x2_input_acceleration
   - 0.0262454345159712 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1803337350399706 x1_input_acceleration
   - 0.2774568241087565 x2_input_acceleration
   - 0.0621244806766908 x3_input_acceleration >= -0.0183404990843331
 CBF_Constraint_h_left: 0.1996662649600294 x1_input_acceleration
   - 0.2774568241087565 x2_input_acceleration
   - 0.0621244806766908 x3_input_acceleration >= -0.0201770894267387
 CBF_Constraint_h_left: - 0.5003337350399706 x1_input_acceleration
   - 0.2774568241087565 x2_input_acceleration
   - 0.0621244806766908 x3_input_acceleration >= -0.0727938966907284
 Relaxed_CLF_constraint: - 0.0203964646879541 x1_input_acceleration
   + 0.0174915963188822 x2_input_acceleration
   - 0.0257921592951019 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1795975104369671 x1_input_acceleration
   - 0.2780905964737221 x2_input_acceleration
   - 0.0612156751035332 x3_input_acceleration >= -0.0183342511201621
 CBF_Constraint_h_left: 0.2004024895630329 x1_input_acceleration
   - 0.2780905964737221 x2_input_acceleration
   - 0.0612156751035332 x3_input_acceleration >= -0.0203107241371383
 CBF_Constraint_h_left: - 0.4996175375976117 x1_input_acceleration
   - 0.2780833247070595 x2_input_acceleration
   - 0.0612168671964287 x3_input_acceleration >= -0.0726738810461445
 Relaxed_CLF_constraint: - 0.0200383659667747 x1_input_acceleration
   + 0.0171783460197307 x2_input_acceleration
   - 0.0253383525549709 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1791637323607936 x1_input_acceleration
   - 0.2794524116875106 x2_input_acceleration
   - 0.0615364230837532 x3_input_acceleration >= -0.0184950061893397
 CBF_Constraint_h_left: 0.2008362676392064 x1_input_acceleration
   - 0.2794524116875106 x2_input_acceleration
   - 0.0615364230837532 x3_input_acceleration >= -0.0205538970407889
 CBF_Constraint_h_left: - 0.4988795374145046 x1_input_acceleration
   - 0.2787129564644271 x2_input_acceleration
   - 0.0603066600527473 x3_input_acceleration >= -0.0725496495496923
 Relaxed_CLF_constraint: - 0.0196793794555434 x1_input_acceleration
   + 0.0168671660243782 x2_input_acceleration
   - 0.024883845029578 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1781605938162319 x1_input_acceleration
   - 0.2793271511390056 x2_input_acceleration
   - 0.0594119389074365 x3_input_acceleration >= -0.0183236582592816
 CBF_Constraint_h_left: 0.2018394061837681 x1_input_acceleration
   - 0.2793271511390056 x2_input_acceleration
   - 0.0594119389074365 x3_input_acceleration >= -0.0205731454341975
 CBF_Constraint_h_left: - 0.4981605938162319 x1_input_acceleration
   - 0.2793271511390056 x2_input_acceleration
   - 0.0594119389074365 x3_input_acceleration >= -0.0724293532698786
 Relaxed_CLF_constraint: - 0.019319907656407 x1_input_acceleration
   + 0.0165570884548502 x2_input_acceleration
   - 0.0244289742716809 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1774619799660817 x1_input_acceleration
   - 0.2799453793825131 x2_input_acceleration
   - 0.0584933773462883 x3_input_acceleration >= -0.0183209112411191
 CBF_Constraint_h_left: 0.2025380200339184 x1_input_acceleration
   - 0.2799453793825131 x2_input_acceleration
   - 0.0584933773462883 x3_input_acceleration >= -0.0207031350475636
 CBF_Constraint_h_left: - 0.4974619799660817 x1_input_acceleration
   - 0.2799453793825131 x2_input_acceleration
   - 0.0584933773462883 x3_input_acceleration >= -0.0723148280356922
 Relaxed_CLF_constraint: - 0.0189603487324305 x1_input_acceleration
   + 0.0162471994727144 x2_input_acceleration
   - 0.0239740446301754 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1769914035479929 x1_input_acceleration
   - 0.2813134201964489 x2_input_acceleration
   - 0.0588016643261202 x3_input_acceleration >= -0.0184802082600085
 CBF_Constraint_h_left: 0.2030081196148489 x1_input_acceleration
   - 0.2813127049407116 x2_input_acceleration
   - 0.0588061942791231 x3_input_acceleration >= -0.0209518257690527
 CBF_Constraint_h_left: - 0.4969918803851511 x1_input_acceleration
   - 0.2813127049407116 x2_input_acceleration
   - 0.0588061942791231 x3_input_acceleration >= -0.0723989839038555
 Relaxed_CLF_constraint: - 0.0186008444436742 x1_input_acceleration
   + 0.015937198211712 x2_input_acceleration
   - 0.0235191478597764 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1762664228903985 x1_input_acceleration
   - 0.2819381123589135 x2_input_acceleration
   - 0.057886931439788 x3_input_acceleration >= -0.0184774619676497
 CBF_Constraint_h_left: 0.2037335771096015 x1_input_acceleration
   - 0.2819381123589135 x2_input_acceleration
   - 0.057886931439788 x3_input_acceleration >= -0.021086841618474
 CBF_Constraint_h_left: - 0.4962664228903985 x1_input_acceleration
   - 0.2819381123589135 x2_input_acceleration
   - 0.057886931439788 x3_input_acceleration >= -0.0722800896301135
 Relaxed_CLF_constraint: - 0.0182409767192472 x1_input_acceleration
   + 0.0156280111766528 x2_input_acceleration
   - 0.023063986788467 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1752775770935064 x1_input_acceleration
   - 0.2818104179956336 x2_input_acceleration
   - 0.0557554132709724 x3_input_acceleration >= -0.0183120017079152
 CBF_Constraint_h_left: 0.2047224229064936 x1_input_acceleration
   - 0.2818104179956336 x2_input_acceleration
   - 0.0557554132709724 x3_input_acceleration >= -0.021109262060149
 CBF_Constraint_h_left: - 0.4952775770935064 x1_input_acceleration
   - 0.2818104179956336 x2_input_acceleration
   - 0.0557554132709724 x3_input_acceleration >= -0.0719564140428762
 Relaxed_CLF_constraint: - 0.0178807834808353 x1_input_acceleration
   + 0.0153195677470257 x2_input_acceleration
   - 0.0226085792665592 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.174573100201345 x1_input_acceleration
   - 0.2824141501087103 x2_input_acceleration
   - 0.05484199823464 x3_input_acceleration >= -0.0183102910664756
 CBF_Constraint_h_left: 0.205426899798655 x1_input_acceleration
   - 0.2824141501087103 x2_input_acceleration
   - 0.05484199823464 x3_input_acceleration >= -0.02124140202822
 CBF_Constraint_h_left: - 0.494573100201345 x1_input_acceleration
   - 0.2824141501087103 x2_input_acceleration
   - 0.05484199823464 x3_input_acceleration >= -0.0718419870986908
 Relaxed_CLF_constraint: - 0.0175211540588025 x1_input_acceleration
   + 0.015009833877377 x2_input_acceleration
   - 0.0221536002830915 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1741185292211609 x1_input_acceleration
   - 0.283780386771104 x2_input_acceleration
   - 0.0551497728196308 x3_input_acceleration >= -0.0184725168940386
 CBF_Constraint_h_left: 0.2058814707788391 x1_input_acceleration
   - 0.283780386771104 x2_input_acceleration
   - 0.0551497728196308 x3_input_acceleration >= -0.0214899963420181
 CBF_Constraint_h_left: - 0.4941185292211609 x1_input_acceleration
   - 0.283780386771104 x2_input_acceleration
   - 0.0551497728196308 x3_input_acceleration >= -0.0719314815694244
 Relaxed_CLF_constraint: - 0.0171610694201508 x1_input_acceleration
   + 0.0147011519246592 x2_input_acceleration
   - 0.0216982685013376 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.173406416538828 x1_input_acceleration
   - 0.2843991984236474 x2_input_acceleration
   - 0.0542567301249855 x3_input_acceleration >= -0.0184741205311765
 CBF_Constraint_h_left: 0.206593583461172 x1_input_acceleration
   - 0.2843991984236474 x2_input_acceleration
   - 0.0542567301249855 x3_input_acceleration >= -0.0216269013887992
 CBF_Constraint_h_left: - 0.4934064165388281 x1_input_acceleration
   - 0.2843991984236474 x2_input_acceleration
   - 0.0542567301249855 x3_input_acceleration >= -0.071819147177389
 Relaxed_CLF_constraint: - 0.0168007215445606 x1_input_acceleration
   + 0.0143930574005725 x2_input_acceleration
   - 0.0212427468526539 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1724203101466751 x1_input_acceleration
   - 0.2842627679475225 x2_input_acceleration
   - 0.0521117722156352 x3_input_acceleration >= -0.0183124303489292
 CBF_Constraint_h_left: 0.2075796898533249 x1_input_acceleration
   - 0.2842627679475225 x2_input_acceleration
   - 0.0521117722156352 x3_input_acceleration >= -0.021652571421061
 CBF_Constraint_h_left: - 0.4924203101466751 x1_input_acceleration
   - 0.2842627679475225 x2_input_acceleration
   - 0.0521117722156352 x3_input_acceleration >= -0.0714996799723972
 Relaxed_CLF_constraint: - 0.0164404674970437 x1_input_acceleration
   + 0.0140847500975888 x2_input_acceleration
   - 0.0207872951806936 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1719716333958931 x1_input_acceleration
   - 0.2856273837957632 x2_input_acceleration
   - 0.0524233887963667 x3_input_acceleration >= -0.0184763641899397
 CBF_Constraint_h_left: 0.2080283666041069 x1_input_acceleration
   - 0.2856273837957632 x2_input_acceleration
   - 0.0524233887963667 x3_input_acceleration >= -0.02190175384472
 CBF_Constraint_h_left: - 0.4919716333958931 x1_input_acceleration
   - 0.2856273837957632 x2_input_acceleration
   - 0.0524233887963667 x3_input_acceleration >= -0.0715918255332826
 Relaxed_CLF_constraint: - 0.0160802305315648 x1_input_acceleration
   + 0.0137764017147892 x2_input_acceleration
   - 0.0203318578388877 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1709599838746898 x1_input_acceleration
   - 0.2855051283453667 x2_input_acceleration
   - 0.0503054057394616 x3_input_acceleration >= -0.0183177820611376
 CBF_Constraint_h_left: 0.2090400161253102 x1_input_acceleration
   - 0.2855051283453667 x2_input_acceleration
   - 0.0503054057394616 x3_input_acceleration >= -0.0219353851249465
 CBF_Constraint_h_left: - 0.4909704742921703 x1_input_acceleration
   - 0.2855031017874443 x2_input_acceleration
   - 0.0502868090902916 x3_input_acceleration >= -0.0712731977313535
 Relaxed_CLF_constraint: - 0.0157200795227942 x1_input_acceleration
   + 0.0134678614330905 x2_input_acceleration
   - 0.0198764825480755 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1702490484560256 x1_input_acceleration
   - 0.286124332242917 x2_input_acceleration
   - 0.0493938169279388 x3_input_acceleration >= -0.018322905288087
 CBF_Constraint_h_left: 0.2097604882871384 x1_input_acceleration
   - 0.2861218288478364 x2_input_acceleration
   - 0.0493945321836761 x3_input_acceleration >= -0.0220762457998334
 CBF_Constraint_h_left: - 0.4902395117128616 x1_input_acceleration
   - 0.2861218288478364 x2_input_acceleration
   - 0.0493945321836761 x3_input_acceleration >= -0.0711600748993349
 Relaxed_CLF_constraint: - 0.0153598434418801 x1_input_acceleration
   + 0.0131595111818557 x2_input_acceleration
   - 0.0194210457701827 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1698022175993971 x1_input_acceleration
   - 0.2874715236901522 x2_input_acceleration
   - 0.049692982380255 x3_input_acceleration >= -0.0184855156330638
 CBF_Constraint_h_left: 0.2104710100922533 x1_input_acceleration
   - 0.2867419628381014 x2_input_acceleration
   - 0.0484832465098937 x3_input_acceleration >= -0.0222174061334099
 CBF_Constraint_h_left: - 0.4895289899077467 x1_input_acceleration
   - 0.2867419628381014 x2_input_acceleration
   - 0.0484832465098937 x3_input_acceleration >= -0.0710525526011213
 Relaxed_CLF_constraint: - 0.0149998141677406 x1_input_acceleration
   + 0.0128506958842635 x2_input_acceleration
   - 0.0189657605611602 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1690830935454278 x1_input_acceleration
   - 0.2880996004450089 x2_input_acceleration
   - 0.0487962254152964 x3_input_acceleration >= -0.0184928859785615
 CBF_Constraint_h_left: 0.2109169064545722 x1_input_acceleration
   - 0.2880996004450089 x2_input_acceleration
   - 0.0487962254152964 x3_input_acceleration >= -0.0224670982049302
 CBF_Constraint_h_left: - 0.4890764178252129 x1_input_acceleration
   - 0.2880876795160539 x2_input_acceleration
   - 0.0487727411852549 x3_input_acceleration >= -0.0711422584611027
 Relaxed_CLF_constraint: - 0.0146397753035978 x1_input_acceleration
   + 0.0125419027155753 x2_input_acceleration
   - 0.0185104679398869 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.168086916936338 x1_input_acceleration
   - 0.2879718719423139 x2_input_acceleration
   - 0.0466600528003023 x3_input_acceleration >= -0.0183395428006127
 CBF_Constraint_h_left: 0.211913083063662 x1_input_acceleration
   - 0.2879718719423139 x2_input_acceleration
   - 0.0466600528003023 x3_input_acceleration >= -0.0225030285827085
 CBF_Constraint_h_left: - 0.488086916936338 x1_input_acceleration
   - 0.2879718719423139 x2_input_acceleration
   - 0.0466600528003023 x3_input_acceleration >= -0.0708334495104268
 Relaxed_CLF_constraint: - 0.0142794929377736 x1_input_acceleration
   + 0.0122336551695901 x2_input_acceleration
   - 0.0180549981713914 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1676292021904061 x1_input_acceleration
   - 0.2893403544848256 x2_input_acceleration
   - 0.0469641770987565 x3_input_acceleration >= -0.01850575602274
 CBF_Constraint_h_left: 0.2123707978095939 x1_input_acceleration
   - 0.2893403544848256 x2_input_acceleration
   - 0.0469641770987565 x3_input_acceleration >= -0.0227562076065628
 CBF_Constraint_h_left: - 0.4876292021904061 x1_input_acceleration
   - 0.2893403544848256 x2_input_acceleration
   - 0.0469641770987565 x3_input_acceleration >= -0.070926428373205
 Relaxed_CLF_constraint: - 0.0139192669277226 x1_input_acceleration
   + 0.0119252807882878 x2_input_acceleration
   - 0.0175995697810677 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1666411476344904 x1_input_acceleration
   - 0.2892092016095225 x2_input_acceleration
   - 0.0448356357876392 x3_input_acceleration >= -0.0183553671542598
 CBF_Constraint_h_left: 0.2133588523655096 x1_input_acceleration
   - 0.2892092016095225 x2_input_acceleration
   - 0.0448356357876392 x3_input_acceleration >= -0.0227935491037066
 CBF_Constraint_h_left: - 0.4866411476344904 x1_input_acceleration
   - 0.2892092016095225 x2_input_acceleration
   - 0.0448356357876392 x3_input_acceleration >= -0.0706179507757783
 Relaxed_CLF_constraint: - 0.0135592308912199 x1_input_acceleration
   + 0.0116164804521052 x2_input_acceleration
   - 0.0171442797811792 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1659090673685347 x1_input_acceleration
   - 0.289812300551533 x2_input_acceleration
   - 0.0439105595394614 x3_input_acceleration >= -0.0183612813562844
 CBF_Constraint_h_left: 0.2140909326314653 x1_input_acceleration
   - 0.289812300551533 x2_input_acceleration
   - 0.0439105595394614 x3_input_acceleration >= -0.0229385585562628
 CBF_Constraint_h_left: - 0.4859090673685347 x1_input_acceleration
   - 0.289812300551533 x2_input_acceleration
   - 0.0439105595394614 x3_input_acceleration >= -0.07050673213525
 Relaxed_CLF_constraint: - 0.0131993896412987 x1_input_acceleration
   + 0.0113072419819239 x2_input_acceleration
   - 0.0166891326330424 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.165487303107992 x1_input_acceleration
   - 0.2911814388425027 x2_input_acceleration
   - 0.044224339454821 x3_input_acceleration >= -0.0185321175041404
 CBF_Constraint_h_left: 0.214512696892008 x1_input_acceleration
   - 0.2911814388425027 x2_input_acceleration
   - 0.044224339454821 x3_input_acceleration >= -0.023189529913622
 CBF_Constraint_h_left: - 0.4854796737134607 x1_input_acceleration
   - 0.2911919292599832 x2_input_acceleration
   - 0.0442375716859612 x3_input_acceleration >= -0.0706100540009375
 Relaxed_CLF_constraint: - 0.0128397343176679 x1_input_acceleration
   + 0.0109975860520762 x2_input_acceleration
   - 0.0162341213074581 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1647646633835691 x1_input_acceleration
   - 0.2918051908435273 x2_input_acceleration
   - 0.043309584611075 x3_input_acceleration >= -0.018543345955578
 CBF_Constraint_h_left: 0.2152353366164309 x1_input_acceleration
   - 0.2918051908435273 x2_input_acceleration
   - 0.043309584611075 x3_input_acceleration >= -0.0233380599126999
 CBF_Constraint_h_left: - 0.4847646633835691 x1_input_acceleration
   - 0.2918051908435273 x2_input_acceleration
   - 0.043309584611075 x3_input_acceleration >= -0.0705056920969491
 Relaxed_CLF_constraint: - 0.0124793681297729 x1_input_acceleration
   + 0.0106895247488296 x2_input_acceleration
   - 0.0157785916295731 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1637777637891414 x1_input_acceleration
   - 0.2916723489296362 x2_input_acceleration
   - 0.0412042164612658 x3_input_acceleration >= -0.0183984256240225
 CBF_Constraint_h_left: 0.2162222362108586 x1_input_acceleration
   - 0.2916723489296362 x2_input_acceleration
   - 0.0412042164612658 x3_input_acceleration >= -0.0233806505040857
 CBF_Constraint_h_left: - 0.4837777637891414 x1_input_acceleration
   - 0.2916723489296362 x2_input_acceleration
   - 0.0412042164612658 x3_input_acceleration >= -0.0702028678302851
 Relaxed_CLF_constraint: - 0.0121189558996977 x1_input_acceleration
   + 0.0103815689319409 x2_input_acceleration
   - 0.0153230268860284 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1630345907022838 x1_input_acceleration
   - 0.2922951939368439 x2_input_acceleration
   - 0.0402859708888747 x3_input_acceleration >= -0.0184099294036244
 CBF_Constraint_h_left: 0.2169654092977162 x1_input_acceleration
   - 0.2922951939368439 x2_input_acceleration
   - 0.0402859708888747 x3_input_acceleration >= -0.0235333571701905
 CBF_Constraint_h_left: - 0.483042220096815 x1_input_acceleration
   - 0.2922926905417633 x2_input_acceleration
   - 0.0402817985637405 x3_input_acceleration >= -0.0700968566589332
 Relaxed_CLF_constraint: - 0.0117588134480658 x1_input_acceleration
   + 0.0100730074512863 x2_input_acceleration
   - 0.0148676591924537 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1625771989470284 x1_input_acceleration
   - 0.2936647623408157 x2_input_acceleration
   - 0.0405750821538594 x3_input_acceleration >= -0.0185791688874855
 CBF_Constraint_h_left: 0.2173946676606376 x1_input_acceleration
   - 0.2936536758768875 x2_input_acceleration
   - 0.0405757974095967 x3_input_acceleration >= -0.0237848295546779
 CBF_Constraint_h_left: - 0.4826053323393624 x1_input_acceleration
   - 0.2936536758768875 x2_input_acceleration
   - 0.0405757974095967 x3_input_acceleration >= -0.0701966958734548
 Relaxed_CLF_constraint: - 0.0113985105147741 x1_input_acceleration
   + 0.00976480539499125 x2_input_acceleration
   - 0.0144121748659922 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1616001571431488 x1_input_acceleration
   - 0.2935346684808045 x2_input_acceleration
   - 0.0384674456035975 x3_input_acceleration >= -0.018439239190023
 CBF_Constraint_h_left: 0.2183998428568512 x1_input_acceleration
   - 0.2935346684808045 x2_input_acceleration
   - 0.0384674456035975 x3_input_acceleration >= -0.0238352093328247
 CBF_Constraint_h_left: - 0.4816001571431489 x1_input_acceleration
   - 0.2935346684808045 x2_input_acceleration
   - 0.0384674456035975 x3_input_acceleration >= -0.0698952643329268
 Relaxed_CLF_constraint: - 0.0110382588083909 x1_input_acceleration
   + 0.00945648883012384 x2_input_acceleration
   - 0.0139567276197614 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1611395029619448 x1_input_acceleration
   - 0.2948855097347698 x2_input_acceleration
   - 0.0387704562281551 x3_input_acceleration >= -0.0186066378856242
 CBF_Constraint_h_left: 0.2188604970380552 x1_input_acceleration
   - 0.2948855097347698 x2_input_acceleration
   - 0.0387704562281551 x3_input_acceleration >= -0.0240901323228547
 CBF_Constraint_h_left: - 0.4811395029619449 x1_input_acceleration
   - 0.2948855097347698 x2_input_acceleration
   - 0.0387704562281551 x3_input_acceleration >= -0.0699889583595354
 Relaxed_CLF_constraint: - 0.0106787088479159 x1_input_acceleration
   + 0.00914656388490387 x2_input_acceleration
   - 0.0135018152760954 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1604419221566425 x1_input_acceleration
   - 0.2955053870097029 x2_input_acceleration
   - 0.0378450815142463 x3_input_acceleration >= -0.0186243235829731
 CBF_Constraint_h_left: 0.2198389349295391 x1_input_acceleration
   - 0.2947715346232282 x2_input_acceleration
   - 0.0366534654558966 x3_input_acceleration >= -0.0241407228662534
 CBF_Constraint_h_left: - 0.4801610650704609 x1_input_acceleration
   - 0.2947715346232282 x2_input_acceleration
   - 0.0366534654558966 x3_input_acceleration >= -0.0696970956409148
 Relaxed_CLF_constraint: - 0.0103187127720469 x1_input_acceleration
   + 0.00883763852639852 x2_input_acceleration
   - 0.0130465784997379 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1594397357976897 x1_input_acceleration
   - 0.295376610667661 x2_input_acceleration
   - 0.0357040195918633 x3_input_acceleration >= -0.018485787123942
 CBF_Constraint_h_left: 0.2202789302789705 x1_input_acceleration
   - 0.2961211918901951 x2_input_acceleration
   - 0.0369339018321587 x3_input_acceleration >= -0.0243936701289662
 CBF_Constraint_h_left: - 0.4797210697210295 x1_input_acceleration
   - 0.2961211918901951 x2_input_acceleration
   - 0.0369339018321587 x3_input_acceleration >= -0.0697960445313265
 Relaxed_CLF_constraint: - 0.00995828655424041 x1_input_acceleration
   + 0.00852973608615237 x2_input_acceleration
   - 0.0125909886586941 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1587253900487705 x1_input_acceleration
   - 0.2960033055190443 x2_input_acceleration
   - 0.0348310104955676 x3_input_acceleration >= -0.0185062264041193
 CBF_Constraint_h_left: 0.2212746099512296 x1_input_acceleration
   - 0.2960033055190443 x2_input_acceleration
   - 0.0348310104955676 x3_input_acceleration >= -0.0244484022948529
 CBF_Constraint_h_left: - 0.4787253900487705 x1_input_acceleration
   - 0.2960033055190443 x2_input_acceleration
   - 0.0348310104955676 x3_input_acceleration >= -0.0695022888119226
 Relaxed_CLF_constraint: - 0.00959825265683159 x1_input_acceleration
   + 0.00822091861346363 x2_input_acceleration
   - 0.012135708647442 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1582677973644803 x1_input_acceleration
   - 0.297369523300627 x2_input_acceleration
   - 0.0351273398808223 x3_input_acceleration >= -0.0186778147694373
 CBF_Constraint_h_left: 0.2217274342639377 x1_input_acceleration
   - 0.2973343565602096 x2_input_acceleration
   - 0.0351130347660762 x3_input_acceleration >= -0.0247009249767066
 CBF_Constraint_h_left: - 0.4782725657360624 x1_input_acceleration
   - 0.2973343565602096 x2_input_acceleration
   - 0.0351130347660762 x3_input_acceleration >= -0.0695963229843284
 Relaxed_CLF_constraint: - 0.00923784925902246 x1_input_acceleration
   + 0.00791296135688857 x2_input_acceleration
   - 0.0116801378931394 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1572726510682549 x1_input_acceleration
   - 0.2972332393467405 x2_input_acceleration
   - 0.0329864377777762 x3_input_acceleration >= -0.0185425976059652
 CBF_Constraint_h_left: 0.2227249647459541 x1_input_acceleration
   - 0.297230974370239 x2_input_acceleration
   - 0.0329915637772269 x3_input_acceleration >= -0.0247602763316588
 CBF_Constraint_h_left: - 0.4772750352540459 x1_input_acceleration
   - 0.297230974370239 x2_input_acceleration
   - 0.0329915637772269 x3_input_acceleration >= -0.0693065386705748
 Relaxed_CLF_constraint: - 0.00887760521247224 x1_input_acceleration
   + 0.00760464039743047 x2_input_acceleration
   - 0.0112246875664088 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1565504331999786 x1_input_acceleration
   - 0.2978396796282921 x2_input_acceleration
   - 0.0320643482719236 x3_input_acceleration >= -0.018561158831572
 CBF_Constraint_h_left: 0.2234476594513886 x1_input_acceleration
   - 0.2978502892550621 x2_input_acceleration
   - 0.0320756731544309 x3_input_acceleration >= -0.0249181250329844
 CBF_Constraint_h_left: - 0.4765523405486114 x1_input_acceleration
   - 0.2978502892550621 x2_input_acceleration
   - 0.0320756731544309 x3_input_acceleration >= -0.0692114442249984
 Relaxed_CLF_constraint: - 0.00851745 x1_input_acceleration
   + 0.00729611544326969 x2_input_acceleration
   - 0.0107693052547362 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1560989178876736 x1_input_acceleration
   - 0.2992102190661838 x2_input_acceleration
   - 0.0324092565850322 x3_input_acceleration >= -0.0187359968179327
 CBF_Constraint_h_left: 0.2239010821123264 x1_input_acceleration
   - 0.2992102190661838 x2_input_acceleration
   - 0.0324092565850322 x3_input_acceleration >= -0.0251772024192747
 CBF_Constraint_h_left: - 0.4760989178876736 x1_input_acceleration
   - 0.2992102190661838 x2_input_acceleration
   - 0.0324092565850322 x3_input_acceleration >= -0.0693118236799605
 Relaxed_CLF_constraint: - 0.00815746263646383 x1_input_acceleration
   + 0.00698718945143562 x2_input_acceleration
   - 0.0103140628971609 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1551252469691786 x1_input_acceleration
   - 0.2990706613950429 x2_input_acceleration
   - 0.0302836428924582 x3_input_acceleration >= -0.0186060504453387
 CBF_Constraint_h_left: 0.2246215524998155 x1_input_acceleration
   - 0.2998269251279531 x2_input_acceleration
   - 0.0314983855529807 x3_input_acceleration >= -0.0253357937928887
 CBF_Constraint_h_left: - 0.4753784475001845 x1_input_acceleration
   - 0.2998269251279531 x2_input_acceleration
   - 0.0314983855529807 x3_input_acceleration >= -0.0692182504179533
 Relaxed_CLF_constraint: - 0.00779722744271927 x1_input_acceleration
   + 0.006678836420551 x2_input_acceleration
   - 0.00985862738113508 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.154396626437811 x1_input_acceleration
   - 0.2996921357322329 x2_input_acceleration
   - 0.0293460227100464 x3_input_acceleration >= -0.0186287208810057
 CBF_Constraint_h_left: 0.225603373562189 x1_input_acceleration
   - 0.2996921357322329 x2_input_acceleration
   - 0.0293460227100464 x3_input_acceleration >= -0.0253933618578216
 CBF_Constraint_h_left: - 0.4743966264378111 x1_input_acceleration
   - 0.2996921357322329 x2_input_acceleration
   - 0.0293460227100464 x3_input_acceleration >= -0.0689321811110554
 Relaxed_CLF_constraint: - 0.00743697029288021 x1_input_acceleration
   + 0.00637054304239182 x2_input_acceleration
   - 0.00940316873035887 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1536773204507141 x1_input_acceleration
   - 0.3003144423677686 x2_input_acceleration
   - 0.0284417437588053 x3_input_acceleration >= -0.0186536039759042
 CBF_Constraint_h_left: 0.2263226795492859 x1_input_acceleration
   - 0.3003144423677686 x2_input_acceleration
   - 0.0284417437588053 x3_input_acceleration >= -0.0255549130902685
 CBF_Constraint_h_left: - 0.4736773204507141 x1_input_acceleration
   - 0.3003144423677686 x2_input_acceleration
   - 0.0284417437588053 x3_input_acceleration >= -0.0688419752480185
 Relaxed_CLF_constraint: - 0.00707684046217355 x1_input_acceleration
   + 0.00606195272434928 x2_input_acceleration
   - 0.00894781060392044 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1532376104564131 x1_input_acceleration
   - 0.3016770499246524 x2_input_acceleration
   - 0.0287753385459656 x3_input_acceleration >= -0.0188297069545169
 CBF_Constraint_h_left: 0.2267871850758135 x1_input_acceleration
   - 0.3016809838312076 x2_input_acceleration
   - 0.0287343305503602 x3_input_acceleration >= -0.0258173762680385
 CBF_Constraint_h_left: - 0.4732128149241865 x1_input_acceleration
   - 0.3016809838312076 x2_input_acceleration
   - 0.0287343305503602 x3_input_acceleration >= -0.0689418614915038
 Relaxed_CLF_constraint: - 0.00671655692193221 x1_input_acceleration
   + 0.00575371441755657 x2_input_acceleration
   - 0.00849233550604554 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1525090643772231 x1_input_acceleration
   - 0.3022869020238454 x2_input_acceleration
   - 0.0278377810995321 x3_input_acceleration >= -0.0188528319772338
 CBF_Constraint_h_left: 0.2274909356227769 x1_input_acceleration
   - 0.3022869020238454 x2_input_acceleration
   - 0.0278377810995321 x3_input_acceleration >= -0.0259761097455615
 CBF_Constraint_h_left: - 0.4725090643772231 x1_input_acceleration
   - 0.3022869020238454 x2_input_acceleration
   - 0.0278377810995321 x3_input_acceleration >= -0.0688542822775896
 Relaxed_CLF_constraint: - 0.00635633699818194 x1_input_acceleration
   + 0.00544533129856306 x2_input_acceleration
   - 0.00803690822325843 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1515211523377956 x1_input_acceleration
   - 0.3021595228897975 x2_input_acceleration
   - 0.0257087533478541 x3_input_acceleration >= -0.0187299942193661
 CBF_Constraint_h_left: 0.2284716951048314 x1_input_acceleration
   - 0.3021672714936183 x2_input_acceleration
   - 0.0257222239975733 x3_input_acceleration >= -0.0260415020583386
 CBF_Constraint_h_left: - 0.4715283048951686 x1_input_acceleration
   - 0.3021672714936183 x2_input_acceleration
   - 0.0257222239975733 x3_input_acceleration >= -0.0685764087716476
 Relaxed_CLF_constraint: - 0.00599613380134414 x1_input_acceleration
   + 0.00513690906825426 x2_input_acceleration
   - 0.00758149421237918 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.151060793896901 x1_input_acceleration
   - 0.3035329038920178 x2_input_acceleration
   - 0.026001651077439 x3_input_acceleration >= -0.0189069182641589
 CBF_Constraint_h_left: 0.228939206103099 x1_input_acceleration
   - 0.3035329038920178 x2_input_acceleration
   - 0.026001651077439 x3_input_acceleration >= -0.0263053674237478
 CBF_Constraint_h_left: - 0.4710736685001725 x1_input_acceleration
   - 0.3035388643564954 x2_input_acceleration
   - 0.0260409901429908 x3_input_acceleration >= -0.0686810941241671
 Relaxed_CLF_constraint: - 0.00563673171102383 x1_input_acceleration
   + 0.00482662189890071 x2_input_acceleration
   - 0.00712671102532228 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1503491479225633 x1_input_acceleration
   - 0.3041447763532616 x2_input_acceleration
   - 0.0251191938116251 x3_input_acceleration >= -0.0189349712904406
 CBF_Constraint_h_left: 0.2296508520774367 x1_input_acceleration
   - 0.3041447763532616 x2_input_acceleration
   - 0.0251191938116251 x3_input_acceleration >= -0.0264686331851536
 CBF_Constraint_h_left: - 0.4700802117653367 x1_input_acceleration
   - 0.3034100895017602 x2_input_acceleration
   - 0.0238880002691446 x3_input_acceleration >= -0.0684009311154171
 Relaxed_CLF_constraint: - 0.00527685560801022 x1_input_acceleration
   + 0.00451746701746092 x2_input_acceleration
   - 0.00667153490553786 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1496296960287171 x1_input_acceleration
   - 0.3047745833671265 x2_input_acceleration
   - 0.0242144313640976 x3_input_acceleration >= -0.0189657328216346
 CBF_Constraint_h_left: 0.2303703039712829 x1_input_acceleration
   - 0.3047745833671265 x2_input_acceleration
   - 0.0242144313640976 x3_input_acceleration >= -0.0266360905761784
 CBF_Constraint_h_left: - 0.4696296960287171 x1_input_acceleration
   - 0.3047745833671265 x2_input_acceleration
   - 0.0242144313640976 x3_input_acceleration >= -0.0685064841862293
 Relaxed_CLF_constraint: - 0.00491641440534982 x1_input_acceleration
   + 0.004209596858612 x2_input_acceleration
   - 0.00621593503095623 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1486361123632451 x1_input_acceleration
   - 0.3046503252125472 x2_input_acceleration
   - 0.0220539822781213 x3_input_acceleration >= -0.0188477231712234
 CBF_Constraint_h_left: 0.2313638876367549 x1_input_acceleration
   - 0.3046503252125472 x2_input_acceleration
   - 0.0220539822781213 x3_input_acceleration >= -0.0267068618222069
 CBF_Constraint_h_left: - 0.4686361123632451 x1_input_acceleration
   - 0.3046503252125472 x2_input_acceleration
   - 0.0220539822781213 x3_input_acceleration >= -0.0682295011493426
 Relaxed_CLF_constraint: - 0.00455647483701813 x1_input_acceleration
   + 0.00390056781288528 x2_input_acceleration
   - 0.00576072479308284 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.147923715572033 x1_input_acceleration
   - 0.3052641332808914 x2_input_acceleration
   - 0.0211680197154455 x3_input_acceleration >= -0.0188789254387608
 CBF_Constraint_h_left: 0.232076284427967 x1_input_acceleration
   - 0.3052641332808914 x2_input_acceleration
   - 0.0211680197154455 x3_input_acceleration >= -0.0268734194800745
 CBF_Constraint_h_left: - 0.467923715572033 x1_input_acceleration
   - 0.3052641332808914 x2_input_acceleration
   - 0.0211680197154455 x3_input_acceleration >= -0.0681467199302861
 Relaxed_CLF_constraint: - 0.00419646174414648 x1_input_acceleration
   + 0.00359169682543559 x2_input_acceleration
   - 0.00530546495492119 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1474889933034048 x1_input_acceleration
   - 0.3066233066169881 x2_input_acceleration
   - 0.0214744353645379 x3_input_acceleration >= -0.0190580016701533
 CBF_Constraint_h_left: 0.2325319875315561 x1_input_acceleration
   - 0.3066202071754598 x2_input_acceleration
   - 0.0214584613197381 x3_input_acceleration >= -0.0271368855589771
 CBF_Constraint_h_left: - 0.4674680124684439 x1_input_acceleration
   - 0.3066202071754598 x2_input_acceleration
   - 0.0214584613197381 x3_input_acceleration >= -0.0682506899229324
 Relaxed_CLF_constraint: - 0.00383652629952963 x1_input_acceleration
   + 0.00328261262931107 x2_input_acceleration
   - 0.00485028817024501 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1467325365648994 x1_input_acceleration
   - 0.3072312375294355 x2_input_acceleration
   - 0.0205320093202319 x3_input_acceleration >= -0.019085758501841
 CBF_Constraint_h_left: 0.2332674634351006 x1_input_acceleration
   - 0.3072312375294355 x2_input_acceleration
   - 0.0205320093202319 x3_input_acceleration >= -0.0273065765545101
 CBF_Constraint_h_left: - 0.4667325365648994 x1_input_acceleration
   - 0.3072312375294355 x2_input_acceleration
   - 0.0205320093202319 x3_input_acceleration >= -0.0681629643522249
 Relaxed_CLF_constraint: - 0.00347641774228866 x1_input_acceleration
   + 0.0029739980107949 x2_input_acceleration
   - 0.00439493 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1460183931768179 x1_input_acceleration
   - 0.3078635980635427 x2_input_acceleration
   - 0.0196493606507233 x3_input_acceleration >= -0.0191218658831381
 CBF_Constraint_h_left: 0.2339816068231821 x1_input_acceleration
   - 0.3078635980635427 x2_input_acceleration
   - 0.0196493606507233 x3_input_acceleration >= -0.0274783711795427
 CBF_Constraint_h_left: - 0.4660183931768179 x1_input_acceleration
   - 0.3078635980635427 x2_input_acceleration
   - 0.0196493606507233 x3_input_acceleration >= -0.068084808791429
 Relaxed_CLF_constraint: - 0.00311624660671955 x1_input_acceleration
   + 0.00266550674291732 x2_input_acceleration
   - 0.00393953895268097 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1450340646285818 x1_input_acceleration
   - 0.3077376662908715 x2_input_acceleration
   - 0.017529071742678 x3_input_acceleration >= -0.0190111548782498
 CBF_Constraint_h_left: 0.2349659353714182 x1_input_acceleration
   - 0.3077376662908715 x2_input_acceleration
   - 0.017529071742678 x3_input_acceleration >= -0.0275546825988193
 CBF_Constraint_h_left: - 0.4650340646285818 x1_input_acceleration
   - 0.3077376662908715 x2_input_acceleration
   - 0.017529071742678 x3_input_acceleration >= -0.0678166052188229
 Relaxed_CLF_constraint: - 0.00275568938826556 x1_input_acceleration
   + 0.00235791055268431 x2_input_acceleration
   - 0.00348384520132555 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1445773521627058 x1_input_acceleration
   - 0.3090989143728651 x2_input_acceleration
   - 0.0178199189483086 x3_input_acceleration >= -0.0191905747840468
 CBF_Constraint_h_left: 0.2354226478372942 x1_input_acceleration
   - 0.3090989143728651 x2_input_acceleration
   - 0.0178199189483086 x3_input_acceleration >= -0.0278208778731327
 CBF_Constraint_h_left: - 0.4645773521627058 x1_input_acceleration
   - 0.3090989143728651 x2_input_acceleration
   - 0.0178199189483086 x3_input_acceleration >= -0.0679229511300797
 Relaxed_CLF_constraint: - 0.00239548768108441 x1_input_acceleration
   + 0.00204945791366506 x2_input_acceleration
   - 0.0030284539848049 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1436169361115285 x1_input_acceleration
   - 0.3089632564326998 x2_input_acceleration
   - 0.0157121517607768 x3_input_acceleration >= -0.0190827474691287
 CBF_Constraint_h_left: 0.2363830638884715 x1_input_acceleration
   - 0.3089632564326998 x2_input_acceleration
   - 0.0157121517607768 x3_input_acceleration >= -0.0278955296079383
 CBF_Constraint_h_left: - 0.4636169361115285 x1_input_acceleration
   - 0.3089632564326998 x2_input_acceleration
   - 0.0157121517607768 x3_input_acceleration >= -0.0676614572469733
 Relaxed_CLF_constraint: - 0.00203544261936295 x1_input_acceleration
   + 0.00174064513341199 x2_input_acceleration
   - 0.00257317983851828 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1431475186863404 x1_input_acceleration
   - 0.3103359954117444 x2_input_acceleration
   - 0.0159919733363484 x3_input_acceleration >= -0.0192638463413612
 CBF_Constraint_h_left: 0.2368524813136596 x1_input_acceleration
   - 0.3103359954117444 x2_input_acceleration
   - 0.0159919733363484 x3_input_acceleration >= -0.0281658177909565
 CBF_Constraint_h_left: - 0.4631475186863404 x1_input_acceleration
   - 0.3103359954117444 x2_input_acceleration
   - 0.0159919733363484 x3_input_acceleration >= -0.0677674493311757
 Relaxed_CLF_constraint: - 0.00167508731558239 x1_input_acceleration
   + 0.00143258632427912 x2_input_acceleration
   - 0.00211764022499793 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1421575822266462 x1_input_acceleration
   - 0.3102022567466687 x2_input_acceleration
   - 0.0138934866221268 x3_input_acceleration >= -0.0191568118114428
 CBF_Constraint_h_left: 0.2378424177733539 x1_input_acceleration
   - 0.3102022567466687 x2_input_acceleration
   - 0.0138934866221268 x3_input_acceleration >= -0.02824687118838
 CBF_Constraint_h_left: - 0.4621575822266462 x1_input_acceleration
   - 0.3102022567466687 x2_input_acceleration
   - 0.0138934866221268 x3_input_acceleration >= -0.0675020249677061
 Relaxed_CLF_constraint: - 0.00131482558292764 x1_input_acceleration
   + 0.00112430402260066 x2_input_acceleration
   - 0.00166217833913955 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1417001127953393 x1_input_acceleration
   - 0.3115503696748514 x2_input_acceleration
   - 0.0141741569293712 x3_input_acceleration >= -0.0193358653838517
 CBF_Constraint_h_left: 0.2382998872046607 x1_input_acceleration
   - 0.3115503696748514 x2_input_acceleration
   - 0.0141741569293712 x3_input_acceleration >= -0.0285128439527373
 CBF_Constraint_h_left: - 0.4617001127953393 x1_input_acceleration
   - 0.3115503696748514 x2_input_acceleration
   - 0.0141741569293712 x3_input_acceleration >= -0.067607883431106
 Relaxed_CLF_constraint: - 9.5448381161014617e-04 x1_input_acceleration
   + 0.00081622 x2_input_acceleration
   - 0.00120664585894216 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1410744816657333 x1_input_acceleration
   - 0.3120982886732591 x2_input_acceleration
   - 0.0133922541428558 x3_input_acceleration >= -0.0193716759102648
 CBF_Constraint_h_left: 0.2389255183342667 x1_input_acceleration
   - 0.3120982886732591 x2_input_acceleration
   - 0.0133922541428558 x3_input_acceleration >= -0.0286675243937755
 CBF_Constraint_h_left: - 0.4610744816657333 x1_input_acceleration
   - 0.3120982886732591 x2_input_acceleration
   - 0.0133922541428558 x3_input_acceleration >= -0.0675435929767821
 Relaxed_CLF_constraint: - 6.4309875828175156e-04 x1_input_acceleration
   + 5.4965155417852074e-04 x2_input_acceleration
   - 8.1295265202481481e-04 x3_input_acceleration + relaxation_CLF
   <= -21.8
 CBF_Constraint_h_left: - 0.1406653392470379 x1_input_acceleration
   - 0.3124446561148353 x2_input_acceleration
   - 0.0128641579804743 x3_input_acceleration >= -0.0193934718401871
 CBF_Constraint_h_left: 0.2393346607529621 x1_input_acceleration
   - 0.3124446561148353 x2_input_acceleration
   - 0.0128641579804743 x3_input_acceleration >= -0.0287670573832499
 CBF_Constraint_h_left: - 0.4606653392470379 x1_input_acceleration
   - 0.3124446561148353 x2_input_acceleration
   - 0.0128641579804743 x3_input_acceleration >= -0.0674999261197132
 Relaxed_CLF_constraint: - 4.3280550303562149e-04 x1_input_acceleration
   + 3.6991132246511071e-04 x2_input_acceleration
   - 5.4711643149078781e-04 x3_input_acceleration + relaxation_CLF
   <= -21.8
 CBF_Constraint_h_left: - 0.1403734190204749 x1_input_acceleration
   - 0.3126721368001004 x2_input_acceleration
   - 0.0124720002664944 x3_input_acceleration >= -0.0194060281723215
 CBF_Constraint_h_left: 0.2396265809795251 x1_input_acceleration
   - 0.3126721368001004 x2_input_acceleration
   - 0.0124720002664944 x3_input_acceleration >= -0.0288350785584313
 CBF_Constraint_h_left: - 0.4603734190204749 x1_input_acceleration
   - 0.3126721368001004 x2_input_acceleration
   - 0.0124720002664944 x3_input_acceleration >= -0.0674657752155975
 Relaxed_CLF_constraint: - 0.000291137 x1_input_acceleration
   + 2.4800514349831992e-04 x2_input_acceleration
   - 3.6778647968271905e-04 x3_input_acceleration + relaxation_CLF
   <= -21.8
 CBF_Constraint_h_left: - 0.1399215130149065 x1_input_acceleration
   - 0.3121123315093524 x2_input_acceleration
   - 0.0110418846988302 x3_input_acceleration >= -0.0192785151255717
 CBF_Constraint_h_left: 0.2400784869850935 x1_input_acceleration
   - 0.3121123315093524 x2_input_acceleration
   - 0.0110418846988302 x3_input_acceleration >= -0.0287934276527395
 CBF_Constraint_h_left: - 0.4599215130149065 x1_input_acceleration
   - 0.3121123315093524 x2_input_acceleration
   - 0.0110418846988302 x3_input_acceleration >= -0.0672659572079568
 Relaxed_CLF_constraint: - 1.9583730274141953e-04 x1_input_acceleration
   + 1.6610759508572936e-04 x2_input_acceleration
   - 2.4716581819558825e-04 x3_input_acceleration + relaxation_CLF
   <= -21.8
 CBF_Constraint_h_left: - 0.1400528266550982 x1_input_acceleration
   - 0.3129637079434469 x2_input_acceleration
   - 0.0120929240666718 x3_input_acceleration >= -0.0194268288890691
 CBF_Constraint_h_left: 0.2399471733449018 x1_input_acceleration
   - 0.3129637079434469 x2_input_acceleration
   - 0.0120929240666718 x3_input_acceleration >= -0.0289167918246004
 CBF_Constraint_h_left: - 0.4600528266550982 x1_input_acceleration
   - 0.3129637079434469 x2_input_acceleration
   - 0.0120929240666718 x3_input_acceleration >= -0.0674352811538848
 Relaxed_CLF_constraint: - 1.3179441580601914e-04 x1_input_acceleration
   + 1.1175631498916427e-04 x2_input_acceleration
   - 1.6632745081636424e-04 x3_input_acceleration + relaxation_CLF
   <= -21.8
 CBF_Constraint_h_left: - 0.1397032238356493 x1_input_acceleration
   - 0.3122977519266739 x2_input_acceleration
   - 0.0107613030290172 x3_input_acceleration >= -0.0192906705628525
 CBF_Constraint_h_left: 0.2402967761643507 x1_input_acceleration
   - 0.3122977519266739 x2_input_acceleration
   - 0.0107613030290172 x3_input_acceleration >= -0.0288470580340792
 CBF_Constraint_h_left: - 0.4597032238356493 x1_input_acceleration
   - 0.3122977519266739 x2_input_acceleration
   - 0.0107613030290172 x3_input_acceleration >= -0.0672431863765564
 Relaxed_CLF_constraint: - 8.76464e-05 x1_input_acceleration
   + 6.96423e-05 x2_input_acceleration
   - 1.0908035514577141e-04 x3_input_acceleration + relaxation_CLF
   <= -21.8
 CBF_Constraint_h_left: - 0.1399179592902671 x1_input_acceleration
   - 0.3130974529342379 x2_input_acceleration
   - 0.0119123966848725 x3_input_acceleration >= -0.0194372388901595
 CBF_Constraint_h_left: 0.2400820407097329 x1_input_acceleration
   - 0.3130974529342379 x2_input_acceleration
   - 0.0119123966848725 x3_input_acceleration >= -0.0289528266250087
 CBF_Constraint_h_left: - 0.4599179592902671 x1_input_acceleration
   - 0.3130974529342379 x2_input_acceleration
   - 0.0119123966848725 x3_input_acceleration >= -0.0674241123766022
 Relaxed_CLF_constraint: - 5.88771e-05 x1_input_acceleration
   + 4.60163e-05 x2_input_acceleration - 7.30239e-05 x3_input_acceleration
   + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1396146774134528 x1_input_acceleration
   - 0.3123851893387706 x2_input_acceleration
   - 0.0106452856367651 x3_input_acceleration >= -0.0192975216934426
 CBF_Constraint_h_left: 0.2403853225865472 x1_input_acceleration
   - 0.3123851893387706 x2_input_acceleration
   - 0.0106452856367651 x3_input_acceleration >= -0.0288707329848865
 CBF_Constraint_h_left: - 0.4596146774134529 x1_input_acceleration
   - 0.3123851893387706 x2_input_acceleration
   - 0.0106452856367651 x3_input_acceleration >= -0.0672358700795951
 Relaxed_CLF_constraint: - 3.90816e-05 x1_input_acceleration
   + 2.84866e-05 x2_input_acceleration - 4.77934e-05 x3_input_acceleration
   + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1398524450328874 x1_input_acceleration
   - 0.3131641367868057 x2_input_acceleration
   - 0.0118330528839709 x3_input_acceleration >= -0.0194426260229142
 CBF_Constraint_h_left: 0.2401475549671127 x1_input_acceleration
   - 0.3131641367868057 x2_input_acceleration
   - 0.0118330528839709 x3_input_acceleration >= -0.0289706614666656
 CBF_Constraint_h_left: - 0.4595844625499772 x1_input_acceleration
   - 0.3124156216577163 x2_input_acceleration
   - 0.0106055548294665 x3_input_acceleration >= -0.0672334691665924
 Relaxed_CLF_constraint: - 2.39742e-05 x1_input_acceleration
   + 1.32704e-05 x2_input_acceleration - 2.7928e-05 x3_input_acceleration
   + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1398246879220843 x1_input_acceleration
   - 0.31317821445248 x2_input_acceleration
   - 0.0117848218401535 x3_input_acceleration >= -0.019442604846489
 CBF_Constraint_h_left: 0.2404394798635602 x1_input_acceleration
   - 0.3124417394616352 x2_input_acceleration
   - 0.0105803311785324 x3_input_acceleration >= -0.0288857318606798
 CBF_Constraint_h_left: - 0.4595605201364398 x1_input_acceleration
   - 0.3124417394616352 x2_input_acceleration
   - 0.0105803311785324 x3_input_acceleration >= -0.0672319139084337
 Relaxed_CLF_constraint: - 1.43872e-05 x1_input_acceleration
   + 3.07255e-06 x2_input_acceleration - 1.51374e-05 x3_input_acceleration
   + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1395402979585066 x1_input_acceleration
   - 0.312457778278312 x2_input_acceleration
   - 0.0105592688853662 x3_input_acceleration >= -0.0193032140300903
 CBF_Constraint_h_left: 0.2404597020414934 x1_input_acceleration
   - 0.312457778278312 x2_input_acceleration
   - 0.0105592688853662 x3_input_acceleration >= -0.028890557417974
 CBF_Constraint_h_left: - 0.4595402979585066 x1_input_acceleration
   - 0.312457778278312 x2_input_acceleration
   - 0.0105592688853662 x3_input_acceleration >= -0.0672296617034513
 Relaxed_CLF_constraint: - 6.42187e-06 x1_input_acceleration
   - 8.16551e-06 x2_input_acceleration - 3.59293e-06 x3_input_acceleration
   + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1398005149934479 x1_input_acceleration
   - 0.3132279710979708 x2_input_acceleration
   - 0.0117510792254556 x3_input_acceleration >= -0.0194485084333869
 CBF_Constraint_h_left: 0.2401994850065521 x1_input_acceleration
   - 0.3132279710979708 x2_input_acceleration
   - 0.0117510792254556 x3_input_acceleration >= -0.0289864105846318
 CBF_Constraint_h_left: - 0.4598048065278717 x1_input_acceleration
   - 0.3132179575176486 x2_input_acceleration
   - 0.0117784973620523 x3_input_acceleration >= -0.0674161705044422
 Relaxed_CLF_constraint: - 6.31795e-07 x1_input_acceleration
   - 1.47724e-05 x2_input_acceleration + 4.29695e-06 x3_input_acceleration
   + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1397809406110905 x1_input_acceleration
   - 0.3132290712537924 x2_input_acceleration
   - 0.0117241473415441 x3_input_acceleration >= -0.0194471545168802
 CBF_Constraint_h_left: 0.2402190593889095 x1_input_acceleration
   - 0.3132290712537924 x2_input_acceleration
   - 0.0117241473415441 x3_input_acceleration >= -0.028988775800773
 CBF_Constraint_h_left: - 0.4595115276167057 x1_input_acceleration
   - 0.3124809137525717 x2_input_acceleration
   - 0.010526809237296 x3_input_acceleration >= -0.0672264947962498
 Relaxed_CLF_constraint: 5.57912e-06 x1_input_acceleration
   - 2.43228e-05 x2_input_acceleration + 1.3531e-05 x3_input_acceleration
   + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1395085770632242 x1_input_acceleration
   - 0.3125046219018681 x2_input_acceleration
   - 0.0105041498098045 x3_input_acceleration >= -0.0193080297368655
 CBF_Constraint_h_left: 0.2404914229367758 x1_input_acceleration
   - 0.3125046219018681 x2_input_acceleration
   - 0.0105041498098045 x3_input_acceleration >= -0.0289014
 CBF_Constraint_h_left: - 0.4595085770632242 x1_input_acceleration
   - 0.3125046219018681 x2_input_acceleration
   - 0.0105041498098045 x3_input_acceleration >= -0.0672294020669814
 Relaxed_CLF_constraint: 9.43858e-06 x1_input_acceleration
   - 3.03952e-05 x2_input_acceleration + 1.93174e-05 x3_input_acceleration
   + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1397813554195005 x1_input_acceleration
   - 0.3132630253432631 x2_input_acceleration
   - 0.0117213693237738 x3_input_acceleration >= -0.0194524852172377
 CBF_Constraint_h_left: 0.2402186445804995 x1_input_acceleration
   - 0.3132630253432631 x2_input_acceleration
   - 0.0117213693237738 x3_input_acceleration >= -0.0289940276875326
 CBF_Constraint_h_left: - 0.4597813554195005 x1_input_acceleration
   - 0.3132630253432631 x2_input_acceleration
   - 0.0117213693237738 x3_input_acceleration >= -0.0674175020843578
 Relaxed_CLF_constraint: 1.39548e-05 x1_input_acceleration
   - 3.86176e-05 x2_input_acceleration + 2.64833e-05 x3_input_acceleration
   + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1394935868493772 x1_input_acceleration
   - 0.312539885612064 x2_input_acceleration
   - 0.0105245452326308 x3_input_acceleration >= -0.0193126017307151
 CBF_Constraint_h_left: 0.2405064131506228 x1_input_acceleration
   - 0.312539885612064 x2_input_acceleration
   - 0.0105245452326308 x3_input_acceleration >= -0.0289088202293334
 CBF_Constraint_h_left: - 0.4597753976098753 x1_input_acceleration
   - 0.3132800560908848 x2_input_acceleration
   - 0.0117098431986342 x3_input_acceleration >= -0.0674187325548409
 Relaxed_CLF_constraint: 1.69337e-05 x1_input_acceleration
   - 4.7133e-05 x2_input_acceleration + 3.22463e-05 x3_input_acceleration
   + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1397744939956375 x1_input_acceleration
   - 0.3132987693464373 x2_input_acceleration
   - 0.011714460526092 x3_input_acceleration >= -0.0194575641577865
 CBF_Constraint_h_left: 0.2402255060043625 x1_input_acceleration
   - 0.3132987693464373 x2_input_acceleration
   - 0.011714460526092 x3_input_acceleration >= -0.0290004102986154
 CBF_Constraint_h_left: - 0.4597744939956375 x1_input_acceleration
   - 0.3132987693464373 x2_input_acceleration
   - 0.011714460526092 x3_input_acceleration >= -0.0674214831970885
 Relaxed_CLF_constraint: 1.8816e-05 x1_input_acceleration
   - 4.97543e-05 x2_input_acceleration + 3.50041e-05 x3_input_acceleration
   + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1394811467575475 x1_input_acceleration
   - 0.3125452655084648 x2_input_acceleration
   - 0.0104719225791702 x3_input_acceleration >= -0.0193122986137654
 CBF_Constraint_h_left: 0.2405188532424525 x1_input_acceleration
   - 0.3125452655084648 x2_input_acceleration
   - 0.0104719225791702 x3_input_acceleration >= -0.0289108807298313
 CBF_Constraint_h_left: - 0.4594811467575475 x1_input_acceleration
   - 0.3125452655084648 x2_input_acceleration
   - 0.0104719225791702 x3_input_acceleration >= -0.067229282094973
 Relaxed_CLF_constraint: 1.98159e-05 x1_input_acceleration
   - 5.13727e-05 x2_input_acceleration + 3.65039e-05 x3_input_acceleration
   + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1397383212675472 x1_input_acceleration
   - 0.3133015377102839 x2_input_acceleration
   - 0.0116874225493215 x3_input_acceleration >= -0.0194553119520368
 CBF_Constraint_h_left: 0.2402616787324528 x1_input_acceleration
   - 0.3133015377102839 x2_input_acceleration
   - 0.0116874225493215 x3_input_acceleration >= -0.0290050309112028
 CBF_Constraint_h_left: - 0.4594712924589535 x1_input_acceleration
   - 0.31256220169649 x2_input_acceleration
   - 0.0104565866347097 x3_input_acceleration >= -0.0672295846818269
 Relaxed_CLF_constraint: 2.4743e-05 x1_input_acceleration
   - 5.98408e-05 x2_input_acceleration + 4.41719e-05 x3_input_acceleration
   + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1397445100403539 x1_input_acceleration
   - 0.3133126974768039 x2_input_acceleration
   - 0.0116945504663022 x3_input_acceleration >= -0.0194575342493047
 CBF_Constraint_h_left: 0.2402626425170191 x1_input_acceleration
   - 0.3133038759893771 x2_input_acceleration
   - 0.0116857289788754 x3_input_acceleration >= -0.0290055030902489
 CBF_Constraint_h_left: - 0.4597373574829809 x1_input_acceleration
   - 0.3133038759893771 x2_input_acceleration
   - 0.0116857289788754 x3_input_acceleration >= -0.0674135782092922
 Relaxed_CLF_constraint: 2.52249e-05 x1_input_acceleration
   - 6.10099e-05 x2_input_acceleration + 4.50187e-05 x3_input_acceleration
   + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1397511569538151 x1_input_acceleration
   - 0.3133085835902403 x2_input_acceleration
   - 0.0116742120425051 x3_input_acceleration >= -0.0194572354120164
 CBF_Constraint_h_left: 0.2402488430461849 x1_input_acceleration
   - 0.3133085835902403 x2_input_acceleration
   - 0.0116742120425051 x3_input_acceleration >= -0.0290045155907916
 CBF_Constraint_h_left: - 0.4597511569538151 x1_input_acceleration
   - 0.3133085835902403 x2_input_acceleration
   - 0.0116742120425051 x3_input_acceleration >= -0.0674174205246268
 Relaxed_CLF_constraint: 2.54777e-05 x1_input_acceleration
   - 6.23504e-05 x2_input_acceleration + 4.57108e-05 x3_input_acceleration
   + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1397272790734562 x1_input_acceleration
   - 0.3133318196332713 x2_input_acceleration
   - 0.0116655877302179 x3_input_acceleration >= -0.0194591569122649
 CBF_Constraint_h_left: 0.2402727209265438 x1_input_acceleration
   - 0.3133318196332713 x2_input_acceleration
   - 0.0116655877302179 x3_input_acceleration >= -0.0290109738883083
 CBF_Constraint_h_left: - 0.4597272790734562 x1_input_acceleration
   - 0.3133318196332713 x2_input_acceleration
   - 0.0116655877302179 x3_input_acceleration >= -0.0674155215640179
 Relaxed_CLF_constraint: 2.93104e-05 x1_input_acceleration
   - 7.15247e-05 x2_input_acceleration + 5.24667e-05 x3_input_acceleration
   + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.139733649956062 x1_input_acceleration
   - 0.3133444708516171 x2_input_acceleration
   - 0.0116680785183066 x3_input_acceleration >= -0.0194615985999081
 CBF_Constraint_h_left: 0.240266350043938 x1_input_acceleration
   - 0.3133444708516171 x2_input_acceleration
   - 0.0116680785183066 x3_input_acceleration >= -0.0290122051082564
 CBF_Constraint_h_left: - 0.459733649956062 x1_input_acceleration
   - 0.3133444708516171 x2_input_acceleration
   - 0.0116680785183066 x3_input_acceleration >= -0.0674189825928781
 Relaxed_CLF_constraint: 3.08934e-05 x1_input_acceleration
   - 7.58237e-05 x2_input_acceleration + 5.54533e-05 x3_input_acceleration
   + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1394697222945713 x1_input_acceleration
   - 0.3125912435059408 x2_input_acceleration
   - 0.0104219190485328 x3_input_acceleration >= -0.0193184263375423
 CBF_Constraint_h_left: 0.2405302777054287 x1_input_acceleration
   - 0.3125912435059408 x2_input_acceleration
   - 0.0104219190485328 x3_input_acceleration >= -0.0289191791015738
 CBF_Constraint_h_left: - 0.4594697222945713 x1_input_acceleration
   - 0.3125912435059408 x2_input_acceleration
   - 0.0104219190485328 x3_input_acceleration >= -0.0672335819046737
 Relaxed_CLF_constraint: 3.19654e-05 x1_input_acceleration
   - 7.79976e-05 x2_input_acceleration + 5.72142e-05 x3_input_acceleration
   + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1394682388870818 x1_input_acceleration
   - 0.3125953945503133 x2_input_acceleration
   - 0.010419092026629 x3_input_acceleration >= -0.0193189569577474
 CBF_Constraint_h_left: 0.2405317611129182 x1_input_acceleration
   - 0.3125953945503133 x2_input_acceleration
   - 0.010419092026629 x3_input_acceleration >= -0.0289199915692018
 CBF_Constraint_h_left: - 0.4594682388870818 x1_input_acceleration
   - 0.3125953945503133 x2_input_acceleration
   - 0.010419092026629 x3_input_acceleration >= -0.0672338751796805
 Relaxed_CLF_constraint: 3.27071e-05 x1_input_acceleration
   - 8.00731e-05 x2_input_acceleration + 5.86277e-05 x3_input_acceleration
   + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1394609250691707 x1_input_acceleration
   - 0.3126046907236406 x2_input_acceleration
   - 0.0104359997739424 x3_input_acceleration >= -0.0193199880937134
 CBF_Constraint_h_left: 0.2405390749308293 x1_input_acceleration
   - 0.3126046907236406 x2_input_acceleration
   - 0.0104359997739424 x3_input_acceleration >= -0.028922412330571
 CBF_Constraint_h_left: - 0.4594609250691707 x1_input_acceleration
   - 0.3126046907236406 x2_input_acceleration
   - 0.0104359997739424 x3_input_acceleration >= -0.0672337361047807
 Relaxed_CLF_constraint: 3.27877e-05 x1_input_acceleration
   - 8.10257e-05 x2_input_acceleration + 5.90549e-05 x3_input_acceleration
   + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1394660913779968 x1_input_acceleration
   - 0.3125970720318629 x2_input_acceleration
   - 0.0104146992027108 x3_input_acceleration >= -0.0193190465116581
 CBF_Constraint_h_left: 0.2405339086220032 x1_input_acceleration
   - 0.3125970720318629 x2_input_acceleration
   - 0.0104146992027108 x3_input_acceleration >= -0.0289204891498387
 CBF_Constraint_h_left: - 0.4594660913779968 x1_input_acceleration
   - 0.3125970720318629 x2_input_acceleration
   - 0.0104146992027108 x3_input_acceleration >= -0.0672336211321376
 Relaxed_CLF_constraint: 3.35424e-05 x1_input_acceleration
   - 8.12695e-05 x2_input_acceleration + 5.98704e-05 x3_input_acceleration
   + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1394604954186356 x1_input_acceleration
   - 0.3126107993556988 x2_input_acceleration
   - 0.0104272791418696 x3_input_acceleration >= -0.0193208674516307
 CBF_Constraint_h_left: 0.2402758136328781 x1_input_acceleration
   - 0.3133527579738629 x2_input_acceleration
   - 0.0116434523138667 x3_input_acceleration >= -0.0290144968821383
 CBF_Constraint_h_left: - 0.4597241863671219 x1_input_acceleration
   - 0.3133527579738629 x2_input_acceleration
   - 0.0116434523138667 x3_input_acceleration >= -0.0674179621106309
 Relaxed_CLF_constraint: 3.56252e-05 x1_input_acceleration
   - 8.52721e-05 x2_input_acceleration + 6.31768e-05 x3_input_acceleration
   + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1394349594435207 x1_input_acceleration
   - 0.31261460826147 x2_input_acceleration
   - 0.0104158335451143 x3_input_acceleration >= -0.019319622700482
 CBF_Constraint_h_left: 0.2405650405564793 x1_input_acceleration
   - 0.31261460826147 x2_input_acceleration
   - 0.0104158335451143 x3_input_acceleration >= -0.0289269804062131
 CBF_Constraint_h_left: - 0.4594349594435207 x1_input_acceleration
   - 0.31261460826147 x2_input_acceleration
   - 0.0104158335451143 x3_input_acceleration >= -0.0672292162114453
 Relaxed_CLF_constraint: 3.88564e-05 x1_input_acceleration
   - 9.03952e-05 x2_input_acceleration + 6.81247e-05 x3_input_acceleration
   + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1397155185633666 x1_input_acceleration
   - 0.3133785133610094 x2_input_acceleration
   - 0.0116365286516934 x3_input_acceleration >= -0.0194654818907121
 CBF_Constraint_h_left: 0.2402844814366334 x1_input_acceleration
   - 0.3133762483845079 x2_input_acceleration
   - 0.0116083952593594 x3_input_acceleration >= -0.0290190149778295
 CBF_Constraint_h_left: - 0.4597155185633666 x1_input_acceleration
   - 0.3133762483845079 x2_input_acceleration
   - 0.0116083952593594 x3_input_acceleration >= -0.0674194464750079
 Relaxed_CLF_constraint: 4.09127e-05 x1_input_acceleration
   - 9.44543e-05 x2_input_acceleration + 7.14666e-05 x3_input_acceleration
   + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1396940921839213 x1_input_acceleration
   - 0.3133664679907324 x2_input_acceleration
   - 0.0115992004896364 x3_input_acceleration >= -0.0194618810260189
 CBF_Constraint_h_left: 0.2402815891210103 x1_input_acceleration
   - 0.3133614612005713 x2_input_acceleration
   - 0.0116135056043825 x3_input_acceleration >= -0.0290163802371745
 CBF_Constraint_h_left: - 0.4597184108789897 x1_input_acceleration
   - 0.3133614612005713 x2_input_acceleration
   - 0.0116135056043825 x3_input_acceleration >= -0.0674178240448209
 Relaxed_CLF_constraint: 4.11355e-05 x1_input_acceleration
   - 9.50477e-05 x2_input_acceleration + 7.18917e-05 x3_input_acceleration
   + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1397221274118301 x1_input_acceleration
   - 0.3133709926524675 x2_input_acceleration
   - 0.0116230947077449 x3_input_acceleration >= -0.0194646870637664
 CBF_Constraint_h_left: 0.2402778725881699 x1_input_acceleration
   - 0.3133709926524675 x2_input_acceleration
   - 0.0116230947077449 x3_input_acceleration >= -0.0290174828555187
 CBF_Constraint_h_left: - 0.4597221274118301 x1_input_acceleration
   - 0.3133709926524675 x2_input_acceleration
   - 0.0116230947077449 x3_input_acceleration >= -0.0674202274496593
 Relaxed_CLF_constraint: 4.21383e-05 x1_input_acceleration
   - 9.86214e-05 x2_input_acceleration + 7.40709e-05 x3_input_acceleration
   + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1394453696164825 x1_input_acceleration
   - 0.3126517702610949 x2_input_acceleration
   - 0.0103945720517703 x3_input_acceleration >= -0.0193260469207533
 CBF_Constraint_h_left: 0.2405546303835175 x1_input_acceleration
   - 0.3126517702610949 x2_input_acceleration
   - 0.0103945720517703 x3_input_acceleration >= -0.0289314266936217
 CBF_Constraint_h_left: - 0.4594453696164825 x1_input_acceleration
   - 0.3126517702610949 x2_input_acceleration
   - 0.0103945720517703 x3_input_acceleration >= -0.0672373060593905
 Relaxed_CLF_constraint: 4.1996e-05 x1_input_acceleration
   - 1.0480387368105859e-04 x2_input_acceleration
   + 7.59541e-05 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.13973257742042 x1_input_acceleration
   - 0.3134034804505075 x2_input_acceleration
   - 0.0116033223420951 x3_input_acceleration >= -0.019470392960105
 CBF_Constraint_h_left: 0.24026742257958 x1_input_acceleration
   - 0.3134034804505075 x2_input_acceleration
   - 0.0116033223420951 x3_input_acceleration >= -0.0290212032502252
 CBF_Constraint_h_left: - 0.4594455214511817 x1_input_acceleration
   - 0.3126662902039254 x2_input_acceleration
   - 0.010396447494683 x3_input_acceleration >= -0.0672396205829858
 Relaxed_CLF_constraint: 4.23969e-05 x1_input_acceleration
   - 1.1063333362171246e-04 x2_input_acceleration
   + 7.8235e-05 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1397060037612159 x1_input_acceleration
   - 0.3134132272684369 x2_input_acceleration
   - 0.0116057433959839 x3_input_acceleration >= -0.0194700779483798
 CBF_Constraint_h_left: 0.240272538566665 x1_input_acceleration
   - 0.3134131080591473 x2_input_acceleration
   - 0.0115978755828735 x3_input_acceleration >= -0.02902329495265
 CBF_Constraint_h_left: - 0.459727461433335 x1_input_acceleration
   - 0.3134131080591473 x2_input_acceleration
   - 0.0115978755828735 x3_input_acceleration >= -0.0674279064543173
 Relaxed_CLF_constraint: 4.47165e-05 x1_input_acceleration
   - 1.1330137072973478e-04 x2_input_acceleration
   + 8.13756e-05 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1397163021184066 x1_input_acceleration
   - 0.3134168846902565 x2_input_acceleration
   - 0.0116226467031948 x3_input_acceleration >= -0.0194714686507437
 CBF_Constraint_h_left: 0.2402836978815934 x1_input_acceleration
   - 0.3134168846902565 x2_input_acceleration
   - 0.0116226467031948 x3_input_acceleration >= -0.0290253712482464
 CBF_Constraint_h_left: - 0.4597163021184066 x1_input_acceleration
   - 0.3134168846902565 x2_input_acceleration
   - 0.0116226467031948 x3_input_acceleration >= -0.0674260769896887
 Relaxed_CLF_constraint: 4.57662e-05 x1_input_acceleration
   - 0.000114832 x2_input_acceleration + 8.29375e-05 x3_input_acceleration
   + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.139446752777003 x1_input_acceleration
   - 0.3126732173373226 x2_input_acceleration
   - 0.0103847031313924 x3_input_acceleration >= -0.0193294449398126
 CBF_Constraint_h_left: 0.240553247222997 x1_input_acceleration
   - 0.3126732173373226 x2_input_acceleration
   - 0.0103847031313924 x3_input_acceleration >= -0.028934561912182
 CBF_Constraint_h_left: - 0.459446752777003 x1_input_acceleration
   - 0.3126732173373226 x2_input_acceleration
   - 0.0103847031313924 x3_input_acceleration >= -0.067240925384133
 Relaxed_CLF_constraint: 4.55959e-05 x1_input_acceleration
   - 1.1564662108443091e-04 x2_input_acceleration
   + 8.30939e-05 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1397070581095823 x1_input_acceleration
   - 0.313421798752981 x2_input_acceleration
   - 0.0115939256813003 x3_input_acceleration >= -0.019471426282973
 CBF_Constraint_h_left: 0.240554248653113 x1_input_acceleration
   - 0.3126738796703394 x2_input_acceleration
   - 0.0103835937644912 x3_input_acceleration >= -0.0289347801481579
 CBF_Constraint_h_left: - 0.459445751346887 x1_input_acceleration
   - 0.3126738796703394 x2_input_acceleration
   - 0.0103835937644912 x3_input_acceleration >= -0.0672407931195683
 Relaxed_CLF_constraint: 4.60966e-05 x1_input_acceleration
   - 1.1597778759286914e-04 x2_input_acceleration
   + 8.36486e-05 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1396838587859768 x1_input_acceleration
   - 0.3134192486963288 x2_input_acceleration
   - 0.0115875195206258 x3_input_acceleration >= -0.0194693691168382
 CBF_Constraint_h_left: 0.2402937298675876 x1_input_acceleration
   - 0.3134230633935944 x2_input_acceleration
   - 0.0115927647293661 x3_input_acceleration >= -0.0290273713686932
 CBF_Constraint_h_left: - 0.4597062701324124 x1_input_acceleration
   - 0.3134230633935944 x2_input_acceleration
   - 0.0115927647293661 x3_input_acceleration >= -0.0674245659150375
 Relaxed_CLF_constraint: 4.64906e-05 x1_input_acceleration
   - 1.1661010789953963e-04 x2_input_acceleration
   + 8.42291e-05 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1394333107379537 x1_input_acceleration
   - 0.3126721219546102 x2_input_acceleration
   - 0.0103606393109673 x3_input_acceleration >= -0.0193282117094693
 CBF_Constraint_h_left: 0.2402829711529154 x1_input_acceleration
   - 0.3134170608050131 x2_input_acceleration
   - 0.011624138570916 x3_input_acceleration >= -0.0290253202068105
 CBF_Constraint_h_left: - 0.4597170288470846 x1_input_acceleration
   - 0.3134170608050131 x2_input_acceleration
   - 0.011624138570916 x3_input_acceleration >= -0.0674262803032901
 Relaxed_CLF_constraint: 4.65949e-05 x1_input_acceleration
   - 0.000117185 x2_input_acceleration + 8.45758e-05 x3_input_acceleration
   + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.13971997476291 x1_input_acceleration
   - 0.313395816634306 x2_input_acceleration
   - 0.0115923153319278 x3_input_acceleration >= -0.0194682477515967
 CBF_Constraint_h_left: 0.24028002523709 x1_input_acceleration
   - 0.313395816634306 x2_input_acceleration
   - 0.0115923153319278 x3_input_acceleration >= -0.0290214525466438
 CBF_Constraint_h_left: - 0.4594329187936717 x1_input_acceleration
   - 0.3126732891303388 x2_input_acceleration
   - 0.0103598104872623 x3_input_acceleration >= -0.0672376295700472
 Relaxed_CLF_constraint: 4.67909e-05 x1_input_acceleration
   - 1.177686801596689e-04 x2_input_acceleration
   + 8.49902e-05 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1397121339985087 x1_input_acceleration
   - 0.3134318046821367 x2_input_acceleration
   - 0.0115949275967113 x3_input_acceleration >= -0.0194733547296728
 CBF_Constraint_h_left: 0.2402878660014913 x1_input_acceleration
   - 0.3134318046821367 x2_input_acceleration
   - 0.0115949275967113 x3_input_acceleration >= -0.0290280492699562
 CBF_Constraint_h_left: - 0.4597121339985087 x1_input_acceleration
   - 0.3134318046821367 x2_input_acceleration
   - 0.0115949275967113 x3_input_acceleration >= -0.0674272961694342
 Relaxed_CLF_constraint: 4.7135e-05 x1_input_acceleration
   - 0.000118239 x2_input_acceleration + 8.54126e-05 x3_input_acceleration
   + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1397136674061789 x1_input_acceleration
   - 0.3134346228523393 x2_input_acceleration
   - 0.0115937117390822 x3_input_acceleration >= -0.0194738964536404
 CBF_Constraint_h_left: 0.2402863325938212 x1_input_acceleration
   - 0.3134346228523393 x2_input_acceleration
   - 0.0115937117390822 x3_input_acceleration >= -0.0290282996464664
 CBF_Constraint_h_left: - 0.4597136674061789 x1_input_acceleration
   - 0.3134346228523393 x2_input_acceleration
   - 0.0115937117390822 x3_input_acceleration >= -0.067428083238629
 Relaxed_CLF_constraint: 4.80372e-05 x1_input_acceleration
   - 0.000123105 x2_input_acceleration + 8.78087e-05 x3_input_acceleration
   + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1396920710445301 x1_input_acceleration
   - 0.3134345765438942 x2_input_acceleration
   - 0.0115725872638957 x3_input_acceleration >= -0.0194722583154852
 CBF_Constraint_h_left: 0.2403079289554699 x1_input_acceleration
   - 0.3134345765438942 x2_input_acceleration
   - 0.0115725872638957 x3_input_acceleration >= -0.0290307648170245
 CBF_Constraint_h_left: - 0.4596920710445301 x1_input_acceleration
   - 0.3134345765438942 x2_input_acceleration
   - 0.0115725872638957 x3_input_acceleration >= -0.06742298968261
 Relaxed_CLF_constraint: 4.81066e-05 x1_input_acceleration
   - 1.2332035736584146e-04 x2_input_acceleration
   + 8.79401e-05 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1397002084518754 x1_input_acceleration
   - 0.3134193737103308 x2_input_acceleration
   - 0.0115662650903959 x3_input_acceleration >= -0.0194704076366537
 CBF_Constraint_h_left: 0.2402997915481246 x1_input_acceleration
   - 0.3134193737103308 x2_input_acceleration
   - 0.0115662650903959 x3_input_acceleration >= -0.0290273680307974
 CBF_Constraint_h_left: - 0.4597106988693558 x1_input_acceleration
   - 0.3134248573376501 x2_input_acceleration
   - 0.0115907029947538 x3_input_acceleration >= -0.0674258530619976
 Relaxed_CLF_constraint: 4.92831e-05 x1_input_acceleration
   - 1.2376556762883073e-04 x2_input_acceleration
   + 8.91938e-05 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1396998239428105 x1_input_acceleration
   - 0.3134344417080916 x2_input_acceleration
   - 0.0115836604410328 x3_input_acceleration >= -0.0194728428119321
 CBF_Constraint_h_left: 0.2403001760571895 x1_input_acceleration
   - 0.3134344417080916 x2_input_acceleration
   - 0.0115836604410328 x3_input_acceleration >= -0.0290298762627981
 CBF_Constraint_h_left: - 0.4596998239428105 x1_input_acceleration
   - 0.3134344417080916 x2_input_acceleration
   - 0.0115836604410328 x3_input_acceleration >= -0.0674248146427818
 Relaxed_CLF_constraint: 4.97137e-05 x1_input_acceleration
   - 1.2378938126755834e-04 x2_input_acceleration
   + 8.96753e-05 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1392730661147976 x1_input_acceleration
   - 0.3126979400712924 x2_input_acceleration
   - 0.00997944637500095 x3_input_acceleration >= -0.0193191445049495
 CBF_Constraint_h_left: 0.2407269338852024 x1_input_acceleration
   - 0.3126979400712924 x2_input_acceleration
   - 0.00997944637500095 x3_input_acceleration >= -0.0289572619431379
 CBF_Constraint_h_left: - 0.4592730661147976 x1_input_acceleration
   - 0.3126979400712924 x2_input_acceleration
   - 0.00997944637500095 x3_input_acceleration >= -0.0672028350833171
 Relaxed_CLF_constraint: 5.01849e-05 x1_input_acceleration
   - 1.2431250009325989e-04 x2_input_acceleration
   + 9.0219e-05 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1394166104373386 x1_input_acceleration
   - 0.3126979768906477 x2_input_acceleration
   - 0.010371161509258 x3_input_acceleration >= -0.0193311442520979
 CBF_Constraint_h_left: 0.2405833895626614 x1_input_acceleration
   - 0.3126979768906477 x2_input_acceleration
   - 0.010371161509258 x3_input_acceleration >= -0.0289419882690036
 CBF_Constraint_h_left: - 0.4594166104373386 x1_input_acceleration
   - 0.3126979768906477 x2_input_acceleration
   - 0.010371161509258 x3_input_acceleration >= -0.0672378019220721
 Relaxed_CLF_constraint: 5.51835e-05 x1_input_acceleration
   - 0.000126894 x2_input_acceleration + 9.58848e-05 x3_input_acceleration
   + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1394339596404719 x1_input_acceleration
   - 0.3126903000395829 x2_input_acceleration
   - 0.0103492384491762 x3_input_acceleration >= -0.0193310398940857
 CBF_Constraint_h_left: 0.2405660403595281 x1_input_acceleration
   - 0.3126903000395829 x2_input_acceleration
   - 0.0103492384491762 x3_input_acceleration >= -0.0289385875623961
 CBF_Constraint_h_left: - 0.4594048725738215 x1_input_acceleration
   - 0.3126960220854813 x2_input_acceleration
   - 0.0103478079377016 x3_input_acceleration >= -0.0672346790754421
 Relaxed_CLF_constraint: 5.50919e-05 x1_input_acceleration
   - 0.000127049 x2_input_acceleration + 9.59387e-05 x3_input_acceleration
   + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1396897649418664 x1_input_acceleration
   - 0.3134524999877992 x2_input_acceleration
   - 0.0115704876056101 x3_input_acceleration >= -0.0194748940903867
 CBF_Constraint_h_left: 0.2405653429377723 x1_input_acceleration
   - 0.3126919447204652 x2_input_acceleration
   - 0.0103493076434519 x3_input_acceleration >= -0.0289387611711333
 CBF_Constraint_h_left: - 0.4594346570622277 x1_input_acceleration
   - 0.3126919447204652 x2_input_acceleration
   - 0.0103493076434519 x3_input_acceleration >= -0.067240891142913
 Relaxed_CLF_constraint: 5.47432e-05 x1_input_acceleration
   - 1.2787133560498987e-04 x2_input_acceleration
   + 9.59041e-05 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.139668656659774 x1_input_acceleration
   - 0.3134403452829047 x2_input_acceleration
   - 0.0115863621146352 x3_input_acceleration >= -0.0194716068728159
 CBF_Constraint_h_left: 0.240331343340226 x1_input_acceleration
   - 0.3134403452829047 x2_input_acceleration
   - 0.0115863621146352 x3_input_acceleration >= -0.0290345621074589
 CBF_Constraint_h_left: - 0.459668656659774 x1_input_acceleration
   - 0.3134403452829047 x2_input_acceleration
   - 0.0115863621146352 x3_input_acceleration >= -0.0674185919383798
 Relaxed_CLF_constraint: 5.50454e-05 x1_input_acceleration
   - 1.2930416839940229e-04 x2_input_acceleration
   + 9.67287e-05 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1396668867366396 x1_input_acceleration
   - 0.3134575896339669 x2_input_acceleration
   - 0.0115719945726618 x3_input_acceleration >= -0.0194741027020578
 CBF_Constraint_h_left: 0.2403331132633604 x1_input_acceleration
   - 0.3134575896339669 x2_input_acceleration
   - 0.0115719945726618 x3_input_acceleration >= -0.0290373942220963
 CBF_Constraint_h_left: - 0.4596668867366396 x1_input_acceleration
   - 0.3134575896339669 x2_input_acceleration
   - 0.0115719945726618 x3_input_acceleration >= -0.0674208045799202
 Relaxed_CLF_constraint: 5.56919e-05 x1_input_acceleration
   - 0.000130595 x2_input_acceleration + 9.78328e-05 x3_input_acceleration
   + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1396946565388366 x1_input_acceleration
   - 0.3134557996436558 x2_input_acceleration
   - 0.0115930675550793 x3_input_acceleration >= -0.019475883652771
 CBF_Constraint_h_left: 0.2403005750895814 x1_input_acceleration
   - 0.313443521086832 x2_input_acceleration
   - 0.0115558742567395 x3_input_acceleration >= -0.0290311863823831
 CBF_Constraint_h_left: - 0.4596994249104187 x1_input_acceleration
   - 0.313443521086832 x2_input_acceleration
   - 0.0115558742567395 x3_input_acceleration >= -0.0674259851010296
 Relaxed_CLF_constraint: 5.56353e-05 x1_input_acceleration
   - 1.3077286107354702e-04 x2_input_acceleration
   + 9.78463e-05 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1394113361197639 x1_input_acceleration
   - 0.3127011267153856 x2_input_acceleration
   - 0.0103364528658076 x3_input_acceleration >= -0.0193310893864041
 CBF_Constraint_h_left: 0.2405886638802361 x1_input_acceleration
   - 0.3127011267153856 x2_input_acceleration
   - 0.0103364528658076 x3_input_acceleration >= -0.028942935523649
 CBF_Constraint_h_left: - 0.4594113361197639 x1_input_acceleration
   - 0.3127011267153856 x2_input_acceleration
   - 0.0103364528658076 x3_input_acceleration >= -0.0672369031655664
 Relaxed_CLF_constraint: 5.56749e-05 x1_input_acceleration
   - 1.3156826339355554e-04 x2_input_acceleration
   + 9.81591e-05 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1404791168073638 x1_input_acceleration
   - 0.3157784754777726 x2_input_acceleration
   - 0.0145684486811594 x3_input_acceleration >= -0.0199156668827547
 CBF_Constraint_h_left: 0.2395390030046479 x1_input_acceleration
   - 0.3157651240373429 x2_input_acceleration
   - 0.0145385271494821 x3_input_acceleration >= -0.0293244790726139
 CBF_Constraint_h_left: - 0.4604609969953521 x1_input_acceleration
   - 0.3157651240373429 x2_input_acceleration
   - 0.0145385271494821 x3_input_acceleration >= -0.0679858280209871
 Relaxed_CLF_constraint: - 3.3683326024935845e-04 x1_input_acceleration
   - 0.0012946737778573 x2_input_acceleration
   - 0.00139669878210798 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1404780030465519 x1_input_acceleration
   - 0.3157849002654549 x2_input_acceleration
   - 0.0145697543105541 x3_input_acceleration >= -0.0199166125790701
 CBF_Constraint_h_left: 0.2395157980703914 x1_input_acceleration
   - 0.3157770324523446 x2_input_acceleration
   - 0.0145622441253124 x3_input_acceleration >= -0.0293237526759177
 CBF_Constraint_h_left: - 0.4604842019296086 x1_input_acceleration
   - 0.3157770324523446 x2_input_acceleration
   - 0.0145622441253124 x3_input_acceleration >= -0.0679932233512807
 Relaxed_CLF_constraint: - 3.3937582137175126e-04 x1_input_acceleration
   - 0.00129395226514328 x2_input_acceleration
   - 0.0013935965041845 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1404887618912465 x1_input_acceleration
   - 0.3157642847396549 x2_input_acceleration
   - 0.0145375992374488 x3_input_acceleration >= -0.0199138793816175
 CBF_Constraint_h_left: 0.2395112381087535 x1_input_acceleration
   - 0.3157642847396549 x2_input_acceleration
   - 0.0145375992374488 x3_input_acceleration >= -0.0293210146222806
 CBF_Constraint_h_left: - 0.4604630126847035 x1_input_acceleration
   - 0.3157732254363712 x2_input_acceleration
   - 0.0145595337467261 x3_input_acceleration >= -0.0679877239940212
 Relaxed_CLF_constraint: - 3.3974845355788652e-04 x1_input_acceleration
   - 0.00129407531507897 x2_input_acceleration
   - 0.001393135384563 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1402130638090981 x1_input_acceleration
   - 0.3150251026133377 x2_input_acceleration
   - 0.0133148353632946 x3_input_acceleration >= -0.0197694508450074
 CBF_Constraint_h_left: 0.2397869361909019 x1_input_acceleration
   - 0.3150251026133377 x2_input_acceleration
   - 0.0133148353632946 x3_input_acceleration >= -0.0292289687212788
 CBF_Constraint_h_left: - 0.4602130638090981 x1_input_acceleration
   - 0.3150251026133377 x2_input_acceleration
   - 0.0133148353632946 x3_input_acceleration >= -0.0678035410544632
 Relaxed_CLF_constraint: - 0.00033899 x1_input_acceleration
   - 0.00129623842138449 x2_input_acceleration
   - 0.00138769426627253 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.140473104864446 x1_input_acceleration
   - 0.3157742554263312 x2_input_acceleration
   - 0.0145046790954316 x3_input_acceleration >= -0.0199141148239932
 CBF_Constraint_h_left: 0.239526895135554 x1_input_acceleration
   - 0.3157742554263312 x2_input_acceleration
   - 0.0145046790954316 x3_input_acceleration >= -0.0293242248997485
 CBF_Constraint_h_left: - 0.460473104864446 x1_input_acceleration
   - 0.3157742554263312 x2_input_acceleration
   - 0.0145046790954316 x3_input_acceleration >= -0.0679898116023046
 Relaxed_CLF_constraint: - 3.4002617184711315e-04 x1_input_acceleration
   - 0.0012961996354679 x2_input_acceleration
   - 0.00138549680098117 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.140476852413344 x1_input_acceleration
   - 0.315775937356213 x2_input_acceleration
   - 0.0145164057413533 x3_input_acceleration >= -0.0199147286782007
 CBF_Constraint_h_left: 0.2395069351232771 x1_input_acceleration
   - 0.315773553170422 x2_input_acceleration
   - 0.0145293995539143 x3_input_acceleration >= -0.0293219030763541
 CBF_Constraint_h_left: - 0.4604930648767229 x1_input_acceleration
   - 0.315773553170422 x2_input_acceleration
   - 0.0145293995539143 x3_input_acceleration >= -0.0679944757832071
 Relaxed_CLF_constraint: - 3.4046943482146652e-04 x1_input_acceleration
   - 0.00129620613538195 x2_input_acceleration
   - 0.00138325389225254 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1402276956826835 x1_input_acceleration
   - 0.3150257341404137 x2_input_acceleration
   - 0.0133204795846988 x3_input_acceleration >= -0.0197706137458871
 CBF_Constraint_h_left: 0.2397723043173165 x1_input_acceleration
   - 0.3150257341404137 x2_input_acceleration
   - 0.0133204795846988 x3_input_acceleration >= -0.0292273515661772
 CBF_Constraint_h_left: - 0.4602276956826835 x1_input_acceleration
   - 0.3150257341404137 x2_input_acceleration
   - 0.0133204795846988 x3_input_acceleration >= -0.0678070450551164
 Relaxed_CLF_constraint: - 3.4106082351956224e-04 x1_input_acceleration
   - 0.00129619655705382 x2_input_acceleration
   - 0.00138276777315383 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1402266906075698 x1_input_acceleration
   - 0.3150257891807646 x2_input_acceleration
   - 0.0133172873420439 x3_input_acceleration >= -0.0197705306874663
 CBF_Constraint_h_left: 0.2397733093924302 x1_input_acceleration
   - 0.3150257891807646 x2_input_acceleration
   - 0.0133172873420439 x3_input_acceleration >= -0.029227459472028
 CBF_Constraint_h_left: - 0.4602266906075698 x1_input_acceleration
   - 0.3150257891807646 x2_input_acceleration
   - 0.0133172873420439 x3_input_acceleration >= -0.0678068011846775
 Relaxed_CLF_constraint: - 3.4055828596268878e-04 x1_input_acceleration
   - 0.00129622407722929 x2_input_acceleration
   - 0.00138117165182636 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1404632467337263 x1_input_acceleration
   - 0.3157725968481628 x2_input_acceleration
   - 0.0144957039248309 x3_input_acceleration >= -0.0199130955088721
 CBF_Constraint_h_left: 0.2395367532662737 x1_input_acceleration
   - 0.3157725968481628 x2_input_acceleration
   - 0.0144957039248309 x3_input_acceleration >= -0.0293250786294641
 CBF_Constraint_h_left: - 0.4604632467337264 x1_input_acceleration
   - 0.3157725968481628 x2_input_acceleration
   - 0.0144957039248309 x3_input_acceleration >= -0.0679872149862683
 Relaxed_CLF_constraint: - 3.3986547806930112e-04 x1_input_acceleration
   - 0.00129900622971502 x2_input_acceleration
   - 0.0013742142861764 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1404821394349404 x1_input_acceleration
   - 0.3158022444901301 x2_input_acceleration
   - 0.0145249030358633 x3_input_acceleration >= -0.0199193154833558
 CBF_Constraint_h_left: 0.2395178605650596 x1_input_acceleration
   - 0.3158022444901301 x2_input_acceleration
   - 0.0145249030358633 x3_input_acceleration >= -0.0293277089907171
 CBF_Constraint_h_left: - 0.4604821394349404 x1_input_acceleration
   - 0.3158022444901301 x2_input_acceleration
   - 0.0145249030358633 x3_input_acceleration >= -0.0679964577929462
 Relaxed_CLF_constraint: - 3.3929824835407518e-04 x1_input_acceleration
   - 0.00130125347065102 x2_input_acceleration
   - 0.00136717735563914 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1402095087607294 x1_input_acceleration
   - 0.3150293372489803 x2_input_acceleration
   - 0.0132888827832064 x3_input_acceleration >= -0.0197696960200207
 CBF_Constraint_h_left: 0.2397904912392706 x1_input_acceleration
   - 0.3150293372489803 x2_input_acceleration
   - 0.0132888827832064 x3_input_acceleration >= -0.0292298893554821
 CBF_Constraint_h_left: - 0.4602095087607294 x1_input_acceleration
   - 0.3150293372489803 x2_input_acceleration
   - 0.0132888827832064 x3_input_acceleration >= -0.0678032174217374
 Relaxed_CLF_constraint: - 3.4126568712747329e-04 x1_input_acceleration
   - 0.00130068032035202 x2_input_acceleration
   - 0.00136506202377482 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1400884242650398 x1_input_acceleration
   - 0.3152838700378771 x2_input_acceleration
   - 0.0130181419623495 x3_input_acceleration >= -0.0197995393348187
 CBF_Constraint_h_left: 0.2399115757349602 x1_input_acceleration
   - 0.3152838700378771 x2_input_acceleration
   - 0.0130181419623495 x3_input_acceleration >= -0.0292827387244611
 CBF_Constraint_h_left: - 0.4600884242650398 x1_input_acceleration
   - 0.3152838700378771 x2_input_acceleration
   - 0.0130181419623495 x3_input_acceleration >= -0.067813687217225
 Relaxed_CLF_constraint: - 2.8120027644087942e-04 x1_input_acceleration
   - 0.00142311873857359 x2_input_acceleration
   - 0.0012436987048686 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1400863219934023 x1_input_acceleration
   - 0.3152918663761568 x2_input_acceleration
   - 0.0130109429419323 x3_input_acceleration >= -0.0198006058122094
 CBF_Constraint_h_left: 0.2399136780065977 x1_input_acceleration
   - 0.3152918663761568 x2_input_acceleration
   - 0.0130109429419323 x3_input_acceleration >= -0.029284204633463
 CBF_Constraint_h_left: - 0.4600863219934023 x1_input_acceleration
   - 0.3152918663761568 x2_input_acceleration
   - 0.0130109429419323 x3_input_acceleration >= -0.0678144173311538
 Relaxed_CLF_constraint: - 2.8277174499224955e-04 x1_input_acceleration
   - 0.00142371944296127 x2_input_acceleration
   - 0.00124099326433161 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1401052836391926 x1_input_acceleration
   - 0.3152870416745366 x2_input_acceleration
   - 0.0130306209291902 x3_input_acceleration >= -0.019801301558325
 CBF_Constraint_h_left: 0.2398947163608074 x1_input_acceleration
   - 0.3152870416745366 x2_input_acceleration
   - 0.0130306209291902 x3_input_acceleration >= -0.0292812976668784
 CBF_Constraint_h_left: - 0.4601052836391926 x1_input_acceleration
   - 0.3152870416745366 x2_input_acceleration
   - 0.0130306209291902 x3_input_acceleration >= -0.0678181469405958
 Relaxed_CLF_constraint: - 2.8414633619794349e-04 x1_input_acceleration
   - 0.00142357206865262 x2_input_acceleration
   - 0.00123956698009799 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1403610133026087 x1_input_acceleration
   - 0.3160229500443792 x2_input_acceleration
   - 0.0142419457457708 x3_input_acceleration >= -0.0199436380071782
 CBF_Constraint_h_left: 0.2396389866973913 x1_input_acceleration
   - 0.3160229500443792 x2_input_acceleration
   - 0.0142419457457708 x3_input_acceleration >= -0.0293750454796825
 CBF_Constraint_h_left: - 0.4603610133026087 x1_input_acceleration
   - 0.3160229500443792 x2_input_acceleration
   - 0.0142419457457708 x3_input_acceleration >= -0.0680014001355956
 Relaxed_CLF_constraint: - 2.8445722808664864e-04 x1_input_acceleration
   - 0.00142400401388887 x2_input_acceleration
   - 0.00123738122096885 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1401024694784782 x1_input_acceleration
   - 0.3152863329412541 x2_input_acceleration
   - 0.013020484999531 x3_input_acceleration >= -0.0198009266807836
 CBF_Constraint_h_left: 0.2398975305215219 x1_input_acceleration
   - 0.3152863329412541 x2_input_acceleration
   - 0.013020484999531 x3_input_acceleration >= -0.0292814574798727
 CBF_Constraint_h_left: - 0.4601024694784782 x1_input_acceleration
   - 0.3152863329412541 x2_input_acceleration
   - 0.013020484999531 x3_input_acceleration >= -0.0678173217973401
 Relaxed_CLF_constraint: - 2.8536186021083099e-04 x1_input_acceleration
   - 0.00142351572523525 x2_input_acceleration
   - 0.00123497585242663 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1403721581812768 x1_input_acceleration
   - 0.3160296993705736 x2_input_acceleration
   - 0.0142067864557683 x3_input_acceleration >= -0.0199452366145311
 CBF_Constraint_h_left: 0.2396278418187232 x1_input_acceleration
   - 0.3160296993705736 x2_input_acceleration
   - 0.0142067864557683 x3_input_acceleration >= -0.0293745265600885
 CBF_Constraint_h_left: - 0.4601032220240502 x1_input_acceleration
   - 0.3152904825660692 x2_input_acceleration
   - 0.0130117133280218 x3_input_acceleration >= -0.067818091999347
 Relaxed_CLF_constraint: - 2.8573813299687112e-04 x1_input_acceleration
   - 0.0014255905376428 x2_input_acceleration
   - 0.00123059 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1400958544645051 x1_input_acceleration
   - 0.3152930537323804 x2_input_acceleration
   - 0.0129984726904332 x3_input_acceleration >= -0.0198013796155784
 CBF_Constraint_h_left: 0.2399041455354949 x1_input_acceleration
   - 0.3152930537323804 x2_input_acceleration
   - 0.0129984726904332 x3_input_acceleration >= -0.0292831672673224
 CBF_Constraint_h_left: - 0.4600958544645051 x1_input_acceleration
   - 0.3152930537323804 x2_input_acceleration
   - 0.0129984726904332 x3_input_acceleration >= -0.0678167163298992
 Relaxed_CLF_constraint: - 2.8825323628095534e-04 x1_input_acceleration
   - 0.00142580323719244 x2_input_acceleration
   - 0.00122873806945975 x3_input_acceleration + relaxation_CLF <= -21.8
 CBF_Constraint_h_left: - 0.1403880430658945 x1_input_acceleration
   - 0.3160398735008678 x2_input_acceleration
   - 0.0142204194627052 x3_input_acceleration >= -0.0199480561520028
 CBF_Constraint_h_left: 0.2398875688115469 x1_input_acceleration
   - 0.3152879013023815 x2_input_acceleration
   - 0.0129933982453591 x3_input_acceleration >= -0.0292803336939844
 CBF_Constraint_h_left: - 0.4601124311884531 x1_input_acceleration
   - 0.3152879013023815 x2_input_acceleration
   - 0.0129933982453591 x3_input_acceleration >= -0.0678196846099429
 Relaxed_CLF_constraint: - 2.9010429661918691e-04 x1_input_acceleration
   - 0.00142567081262879 x2_input_acceleration
   - 0.0012270949165943 x3_input_acceleration + relaxation_CLF <= -21.8
Bounds
 -0.1 <= x1_input_acceleration <= 0.1
 -0.1 <= x2_input_acceleration <= 0.1
 -0.1 <= x3_input_acceleration <= 0.1
 -30 <= relaxation_CLF <= 30
End
