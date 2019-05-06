\ Model CBF_CLF_QP
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
 [ 2 x1_input_acceleration ^2 + 2 x2_input_acceleration ^2
   + 2 x3_input_acceleration ^2 + 2 relaxation_CLF ^2 ] / 2 
Subject To
 CBF_Constraint_obs2: 0.1148517231768088 x1_input_acceleration
   + 0.5751316105670681 x2_input_acceleration
   - 0.1312431212701873 x3_input_acceleration >= -0.4014900558386163
 CBF_Constraint_obs3: 0.4948517231768088 x1_input_acceleration
   + 0.5751316105670681 x2_input_acceleration
   - 0.1312431212701873 x3_input_acceleration >= -0.6910991928565846
 CBF_Constraint_obs1: - 0.2051482768231913 x1_input_acceleration
   + 0.5751316105670681 x2_input_acceleration
   - 0.1312431212701873 x3_input_acceleration >= -0.4376086772971693
 CBF_Constraint_for_table: x3_input_acceleration >= 0.088154785940666
 Relaxed_CLF_constraint: 0.1193658830270029 x1_input_acceleration
   + 0.4435084732446546 x2_input_acceleration
   - 0.0647147780079879 x3_input_acceleration + relaxation_CLF
   <= -20.0860543929453
 CBF_Constraint_obs2: 0.1118517231869808 x1_input_acceleration
   + 0.5721316105698732 x2_input_acceleration
   - 0.128243121290947 x3_input_acceleration >= -0.3953646074395233
 CBF_Constraint_obs3: 0.4918517231869808 x1_input_acceleration
   + 0.5721316105698732 x2_input_acceleration
   - 0.128243121290947 x3_input_acceleration >= -0.6821237444671548
 CBF_Constraint_obs1: - 0.2081482768130192 x1_input_acceleration
   + 0.5721316105698732 x2_input_acceleration
   - 0.128243121290947 x3_input_acceleration >= -0.4338832288899386
 CBF_Constraint_for_table: x3_input_acceleration >= 0.0806547859925652
 Relaxed_CLF_constraint: 0.117865883032089 x1_input_acceleration
   + 0.4420084732460571 x2_input_acceleration
   - 0.0632147780183678 x3_input_acceleration + relaxation_CLF
   <= -20.08530398598566
 CBF_Constraint_obs2: 0.1089332623688543 x1_input_acceleration
   + 0.5691790558743715 x2_input_acceleration
   - 0.1252362071959324 x3_input_acceleration >= -0.3893942011114812
 CBF_Constraint_obs3: 0.4889332623688543 x1_input_acceleration
   + 0.5691790558743715 x2_input_acceleration
   - 0.1252362071959324 x3_input_acceleration >= -0.6733808003618926
 CBF_Constraint_obs1: - 0.2110667376311457 x1_input_acceleration
   + 0.5691790558743715 x2_input_acceleration
   - 0.1252362071959324 x3_input_acceleration >= -0.4302475912163978
 CBF_Constraint_for_table: x3_input_acceleration >= 0.0736068873326349
 Relaxed_CLF_constraint: 0.1163658830459993 x1_input_acceleration
   + 0.4405084732496857 x2_input_acceleration
   - 0.0617147780402574 x3_input_acceleration + relaxation_CLF
   <= -20.08455897902816
 CBF_Constraint_obs2: 0.1059332623690338 x1_input_acceleration
   + 0.56617905587442 x2_input_acceleration
   - 0.1222362071965905 x3_input_acceleration >= -0.3834028371710049
 CBF_Constraint_obs3: 0.4859332623690338 x1_input_acceleration
   + 0.56617905587442 x2_input_acceleration
   - 0.1222362071965905 x3_input_acceleration >= -0.6645394364215869
 CBF_Constraint_obs1: - 0.2140667376309662 x1_input_acceleration
   + 0.56617905587442 x2_input_acceleration
   - 0.1222362071965905 x3_input_acceleration >= -0.4266562272757779
 CBF_Constraint_for_table: x3_input_acceleration >= 0.0661068873342802
 Relaxed_CLF_constraint: 0.1148658830460891 x1_input_acceleration
   + 0.4390084732497099 x2_input_acceleration
   - 0.0602147780405865 x3_input_acceleration + relaxation_CLF
   <= -20.08381937206699
 CBF_Constraint_obs2: 0.1029332623713147 x1_input_acceleration
   + 0.5631790558750158 x2_input_acceleration
   - 0.119236207200089 x3_input_acceleration >= -0.3774789732326733
 CBF_Constraint_obs3: 0.4829332623713147 x1_input_acceleration
   + 0.5631790558750158 x2_input_acceleration
   - 0.119236207200089 x3_input_acceleration >= -0.6557655724854222
 CBF_Constraint_obs1: - 0.2170667376286853 x1_input_acceleration
   + 0.5631790558750158 x2_input_acceleration
   - 0.119236207200089 x3_input_acceleration >= -0.4231323633356215
 CBF_Constraint_for_table: x3_input_acceleration >= 0.0586068873430265
 Relaxed_CLF_constraint: 0.1133658830472295 x1_input_acceleration
   + 0.4375084732500079 x2_input_acceleration
   - 0.0587147780423357 x3_input_acceleration + relaxation_CLF
   <= -20.08308516510608
 CBF_Constraint_obs2: 0.0999332623723848 x1_input_acceleration
   + 0.5601790558752923 x2_input_acceleration
   - 0.1162362072018571 x3_input_acceleration >= -0.3716226092929935
 CBF_Constraint_obs3: 0.4799332623723848 x1_input_acceleration
   + 0.5601790558752923 x2_input_acceleration
   - 0.1162362072018571 x3_input_acceleration >= -0.647059208546759
 CBF_Constraint_obs1: - 0.2200667376276152 x1_input_acceleration
   + 0.5601790558752923 x2_input_acceleration
   - 0.1162362072018571 x3_input_acceleration >= -0.4196759993950856
 CBF_Constraint_for_table: x3_input_acceleration >= 0.0511068873474467
 Relaxed_CLF_constraint: 0.1118658830477646 x1_input_acceleration
   + 0.4360084732501461 x2_input_acceleration
   - 0.0572147780432198 x3_input_acceleration + relaxation_CLF
   <= -20.08235635814501
 CBF_Constraint_obs2: 0.0969332623925255 x1_input_acceleration
   + 0.5571790558805962 x2_input_acceleration
   - 0.1132362072338497 x3_input_acceleration >= -0.3658337453734477
 CBF_Constraint_obs3: 0.4769332623925255 x1_input_acceleration
   + 0.5571790558805962 x2_input_acceleration
   - 0.1132362072338497 x3_input_acceleration >= -0.6384203446463468
 CBF_Constraint_obs1: - 0.2230667376074745 x1_input_acceleration
   + 0.5571790558805962 x2_input_acceleration
   - 0.1132362072338497 x3_input_acceleration >= -0.4162871354594273
 CBF_Constraint_for_table: x3_input_acceleration >= 0.0436068874274282
 Relaxed_CLF_constraint: 0.1103658830578349 x1_input_acceleration
   + 0.4345084732527981 x2_input_acceleration
   - 0.0557147780592161 x3_input_acceleration + relaxation_CLF
   <= -20.08163295118632
 CBF_Constraint_obs2: 0.0939332624001935 x1_input_acceleration
   + 0.5541790558826774 x2_input_acceleration
   - 0.1102362072454469 x3_input_acceleration >= -0.3601123814400254
 CBF_Constraint_obs3: 0.4739332624001935 x1_input_acceleration
   + 0.5541790558826774 x2_input_acceleration
   - 0.1102362072454469 x3_input_acceleration >= -0.6298489807202091
 CBF_Constraint_obs1: - 0.2260667375998066 x1_input_acceleration
   + 0.5541790558826774 x2_input_acceleration
   - 0.1102362072454469 x3_input_acceleration >= -0.4129657715198706
 CBF_Constraint_for_table: x3_input_acceleration >= 0.0361068874564213
 Relaxed_CLF_constraint: 0.1088658830616689 x1_input_acceleration
   + 0.4330084732538386 x2_input_acceleration
   - 0.0542147780650147 x3_input_acceleration + relaxation_CLF
   <= -20.08091494422602
 CBF_Constraint_obs2: 0.0909332624061271 x1_input_acceleration
   + 0.5511790558841976 x2_input_acceleration
   - 0.1072362072541702 x3_input_acceleration >= -0.3544585175043453
 CBF_Constraint_obs3: 0.4709332624061272 x1_input_acceleration
   + 0.5511790558841976 x2_input_acceleration
   - 0.1072362072541702 x3_input_acceleration >= -0.6213451167901662
 CBF_Constraint_obs1: - 0.2290667375938729 x1_input_acceleration
   + 0.5511790558841976 x2_input_acceleration
   - 0.1072362072541702 x3_input_acceleration >= -0.4097119075794436
 CBF_Constraint_for_table: x3_input_acceleration >= 0.0286068874782294
 Relaxed_CLF_constraint: 0.1073658830646358 x1_input_acceleration
   + 0.4315084732545987 x2_input_acceleration
   - 0.0527147780693763 x3_input_acceleration + relaxation_CLF
   <= -20.08020233726547
 CBF_Constraint_obs2: 0.087933262406646 x1_input_acceleration
   + 0.5481790558843345 x2_input_acceleration
   - 0.1042362072551377 x3_input_acceleration >= -0.3488721535633155
 CBF_Constraint_obs3: 0.467933262406646 x1_input_acceleration
   + 0.5481790558843345 x2_input_acceleration
   - 0.1042362072551377 x3_input_acceleration >= -0.6129087528496291
 CBF_Constraint_obs1: - 0.2320667375933541 x1_input_acceleration
   + 0.5481790558843345 x2_input_acceleration
   - 0.1042362072551377 x3_input_acceleration >= -0.4065255436379988
 CBF_Constraint_for_table: x3_input_acceleration >= 0.0211068874806481
 Relaxed_CLF_constraint: 0.1058658830648952 x1_input_acceleration
   + 0.4300084732546672 x2_input_acceleration
   - 0.0512147780698601 x3_input_acceleration + relaxation_CLF
   <= -20.07949513030427
 CBF_Constraint_obs2: 0.0849332624123181 x1_input_acceleration
   + 0.5451790558858156 x2_input_acceleration
   - 0.1012362072616425 x3_input_acceleration >= -0.343353289626589
 CBF_Constraint_obs3: 0.4649332624123181 x1_input_acceleration
   + 0.5451790558858156 x2_input_acceleration
   - 0.1012362072616425 x3_input_acceleration >= -0.6045398889182912
 CBF_Constraint_obs1: - 0.2350667375876819 x1_input_acceleration
   + 0.5451790558858156 x2_input_acceleration
   - 0.1012362072616425 x3_input_acceleration >= -0.4034066796967345
 CBF_Constraint_for_table: x3_input_acceleration >= 0.0136068874969103
 Relaxed_CLF_constraint: 0.1043658830677312 x1_input_acceleration
   + 0.4285084732554078 x2_input_acceleration
   - 0.0497147780731125 x3_input_acceleration + relaxation_CLF
   <= -20.07879332334362
 CBF_Constraint_obs2: 0.0818717504248443 x1_input_acceleration
   + 0.5421176631031961 x2_input_acceleration
   - 0.0982719700578252 x3_input_acceleration >= -0.3378149053314307
 CBF_Constraint_obs3: 0.4618717504248444 x1_input_acceleration
   + 0.5421176631031961 x2_input_acceleration
   - 0.0982719700578252 x3_input_acceleration >= -0.5960930682350327
 CBF_Constraint_obs1: - 0.2380667375817475 x1_input_acceleration
   + 0.5421790558873147 x2_input_acceleration
   - 0.0982362072709599 x3_input_acceleration >= -0.4003553157558244
 CBF_Constraint_for_table: x3_input_acceleration >= 0.00604042834127922
 Relaxed_CLF_constraint: 0.1028658830706985 x1_input_acceleration
   + 0.4270084732561573 x2_input_acceleration
   - 0.0482147780777712 x3_input_acceleration + relaxation_CLF
   <= -20.07809691638302
 CBF_Constraint_obs2: 0.0788717504248488 x1_input_acceleration
   + 0.5391176631031981 x2_input_acceleration
   - 0.0952719700578416 x3_input_acceleration >= -0.3324316949545441
 CBF_Constraint_obs3: 0.4588717504248488 x1_input_acceleration
   + 0.5391176631031981 x2_input_acceleration
   - 0.0952719700578416 x3_input_acceleration >= -0.5878598578581505
 CBF_Constraint_obs1: - 0.2411282495751512 x1_input_acceleration
   + 0.5391176631031981 x2_input_acceleration
   - 0.0952719700578416 x3_input_acceleration >= -0.3973342946146651
 CBF_Constraint_for_table: x3_input_acceleration >= -0.00145957165867974
 Relaxed_CLF_constraint: 0.1013658830707007 x1_input_acceleration
   + 0.4255084732561583 x2_input_acceleration
   - 0.0467147780777794 x3_input_acceleration + relaxation_CLF
   <= -20.07740590942174
 CBF_Constraint_obs2: 0.0758717504314726 x1_input_acceleration
   + 0.5361176631050244 x2_input_acceleration
   - 0.0922719700699791 x3_input_acceleration >= -0.327115984584154
 CBF_Constraint_obs3: 0.4558717504314726 x1_input_acceleration
   + 0.5361176631050244 x2_input_acceleration
   - 0.0922719700699791 x3_input_acceleration >= -0.5796941474940529
 CBF_Constraint_obs1: - 0.2441282495685274 x1_input_acceleration
   + 0.5361176631050244 x2_input_acceleration
   - 0.0922719700699791 x3_input_acceleration >= -0.3944185842389759
 CBF_Constraint_for_table: x3_input_acceleration >= -0.008959571628336
 Relaxed_CLF_constraint: 0.0998658830740126 x1_input_acceleration
   + 0.4240084732570715 x2_input_acceleration
   - 0.0452147780838482 x3_input_acceleration + relaxation_CLF
   <= -20.07672030246125
 CBF_Constraint_obs2: 0.072871750432174 x1_input_acceleration
   + 0.5331176631053312 x2_input_acceleration
   - 0.0892719700714341 x3_input_acceleration >= -0.3218677742079668
 CBF_Constraint_obs3: 0.452871750432174 x1_input_acceleration
   + 0.5331176631053312 x2_input_acceleration
   - 0.0892719700714341 x3_input_acceleration >= -0.5715959371185321
 CBF_Constraint_obs1: - 0.247128249567826 x1_input_acceleration
   + 0.5331176631053312 x2_input_acceleration
   - 0.0892719700714341 x3_input_acceleration >= -0.3915703738622275
 CBF_Constraint_for_table: x3_input_acceleration >= -0.0164595716246985
 Relaxed_CLF_constraint: 0.0983658830743633 x1_input_acceleration
   + 0.4225084732572248 x2_input_acceleration
   - 0.0437147780845757 x3_input_acceleration + relaxation_CLF
   <= -20.07604009550005
 CBF_Constraint_obs2: 0.0698717504328185 x1_input_acceleration
   + 0.5301176631054956 x2_input_acceleration
   - 0.0862719700725769 x3_input_acceleration >= -0.3166870638314767
 CBF_Constraint_obs3: 0.4498717504328185 x1_input_acceleration
   + 0.5301176631054956 x2_input_acceleration
   - 0.0862719700725769 x3_input_acceleration >= -0.5635652267426543
 CBF_Constraint_obs1: - 0.2501282495671815 x1_input_acceleration
   + 0.5301176631054956 x2_input_acceleration
   - 0.0862719700725769 x3_input_acceleration >= -0.388789663485222
 CBF_Constraint_for_table: x3_input_acceleration >= -0.0239595716218413
 Relaxed_CLF_constraint: 0.0968658830746855 x1_input_acceleration
   + 0.4210084732573071 x2_input_acceleration
   - 0.0422147780851471 x3_input_acceleration + relaxation_CLF
   <= -20.07536528853883
 CBF_Constraint_obs2: 0.0668717504375484 x1_input_acceleration
   + 0.5271176631066635 x2_input_acceleration
   - 0.0832719700804985 x3_input_acceleration >= -0.311573853458374
 CBF_Constraint_obs3: 0.4468717504375484 x1_input_acceleration
   + 0.5271176631066635 x2_input_acceleration
   - 0.0832719700804985 x3_input_acceleration >= -0.5556020163740449
 CBF_Constraint_obs1: - 0.2531282495624516 x1_input_acceleration
   + 0.5271176631066635 x2_input_acceleration
   - 0.0832719700804985 x3_input_acceleration >= -0.3860764531083352
 CBF_Constraint_for_table: x3_input_acceleration >= -0.0314595716020373
 Relaxed_CLF_constraint: 0.0953658830770505 x1_input_acceleration
   + 0.419508473257891 x2_input_acceleration
   - 0.0407147780891079 x3_input_acceleration + relaxation_CLF
   <= -20.07469588157803
 CBF_Constraint_obs2: 0.0638717504446658 x1_input_acceleration
   + 0.5241176631088242 x2_input_acceleration
   - 0.0802719700918368 x3_input_acceleration >= -0.3065281430874316
 CBF_Constraint_obs3: 0.4438717504446658 x1_input_acceleration
   + 0.5241176631088242 x2_input_acceleration
   - 0.0802719700918368 x3_input_acceleration >= -0.5477063060098641
 CBF_Constraint_obs1: - 0.2561282495553342 x1_input_acceleration
   + 0.5241176631088242 x2_input_acceleration
   - 0.0802719700918368 x3_input_acceleration >= -0.3834307427316991
 CBF_Constraint_for_table: x3_input_acceleration >= -0.0389595715736917
 Relaxed_CLF_constraint: 0.0938658830806092 x1_input_acceleration
   + 0.4180084732589714 x2_input_acceleration
   - 0.039214778094777 x3_input_acceleration + relaxation_CLF
   <= -20.07403187461753
 CBF_Constraint_obs2: 0.0609342161133311 x1_input_acceleration
   + 0.5214844700930407 x2_input_acceleration
   - 0.0769863446360495 x3_input_acceleration >= -0.3019824106274826
 CBF_Constraint_obs3: 0.4408717504456065 x1_input_acceleration
   + 0.5211176631090929 x2_input_acceleration
   - 0.0772719700938131 x3_input_acceleration >= -0.539878095634293
 CBF_Constraint_obs1: - 0.2591282495543935 x1_input_acceleration
   + 0.5211176631090929 x2_input_acceleration
   - 0.0772719700938131 x3_input_acceleration >= -0.3808525323544816
 CBF_Constraint_for_table: x3_input_acceleration >= -0.0468139211819405
 Relaxed_CLF_constraint: 0.0923658830810795 x1_input_acceleration
   + 0.4165084732591057 x2_input_acceleration
   - 0.0377147780957652 x3_input_acceleration + relaxation_CLF
   <= -20.07337326765632
 CBF_Constraint_obs2: 0.0579342161677998 x1_input_acceleration
   + 0.5184844701078217 x2_input_acceleration
   - 0.073986344703233 x3_input_acceleration >= -0.2970706229356395
 CBF_Constraint_obs3: 0.4379342161677998 x1_input_acceleration
   + 0.5184844701078217 x2_input_acceleration
   - 0.073986344703233 x3_input_acceleration >= -0.5326081282950492
 CBF_Constraint_obs1: - 0.2620657838322002 x1_input_acceleration
   + 0.5184844701078217 x2_input_acceleration
   - 0.073986344703233 x3_input_acceleration >= -0.3787232500013997
 CBF_Constraint_for_table: x3_input_acceleration >= -0.0543139210139818
 Relaxed_CLF_constraint: 0.0908658831083139 x1_input_acceleration
   + 0.4150084732664963 x2_input_acceleration
   - 0.0362147781293569 x3_input_acceleration + relaxation_CLF
   <= -20.0727200607004
 CBF_Constraint_obs2: 0.054934216170283 x1_input_acceleration
   + 0.515484470108571 x2_input_acceleration
   - 0.0709863447078457 x3_input_acceleration >= -0.2922263352054234
 CBF_Constraint_obs3: 0.434934216170283 x1_input_acceleration
   + 0.515484470108571 x2_input_acceleration
   - 0.0709863447078457 x3_input_acceleration >= -0.5249138405671923
 CBF_Constraint_obs1: - 0.265065783829717 x1_input_acceleration
   + 0.515484470108571 x2_input_acceleration
   - 0.0709863447078457 x3_input_acceleration >= -0.376278962269197
 CBF_Constraint_for_table: x3_input_acceleration >= -0.0618139210024498
 Relaxed_CLF_constraint: 0.0893658831095555 x1_input_acceleration
   + 0.4135084732668709 x2_input_acceleration
   - 0.0347147781316633 x3_input_acceleration + relaxation_CLF
   <= -20.07207225373927
 CBF_Constraint_obs2: 0.0519342161703994 x1_input_acceleration
   + 0.512484470108607 x2_input_acceleration
   - 0.0679863447079192 x3_input_acceleration >= -0.2874495474730969
 CBF_Constraint_obs3: 0.4319342161703995 x1_input_acceleration
   + 0.512484470108607 x2_input_acceleration
   - 0.0679863447079192 x3_input_acceleration >= -0.5172870528349762
 CBF_Constraint_obs1: - 0.2680657838296006 x1_input_acceleration
   + 0.512484470108607 x2_input_acceleration
   - 0.0679863447079192 x3_input_acceleration >= -0.3739021745367774
 CBF_Constraint_for_table: x3_input_acceleration >= -0.0693139210022663
 Relaxed_CLF_constraint: 0.0878658831096137 x1_input_acceleration
   + 0.4120084732668889 x2_input_acceleration
   - 0.0332147781317 x3_input_acceleration + relaxation_CLF
   <= -20.07142984677787
 CBF_Constraint_obs2: 0.0489556738425695 x1_input_acceleration
   + 0.5094782712255674 x2_input_acceleration
   - 0.0649721588025499 x3_input_acceleration >= -0.2827326853397958
 CBF_Constraint_obs3: 0.4289342161704504 x1_input_acceleration
   + 0.5094844701086241 x2_input_acceleration
   - 0.0649863447080064 x3_input_acceleration >= -0.509727765102665
 CBF_Constraint_obs1: - 0.2710657838295496 x1_input_acceleration
   + 0.5094844701086241 x2_input_acceleration
   - 0.0649863447080064 x3_input_acceleration >= -0.3715928868043767
 CBF_Constraint_for_table: x3_input_acceleration >= -0.0765147056852757
 Relaxed_CLF_constraint: 0.0863658831096392 x1_input_acceleration
   + 0.4105084732668974 x2_input_acceleration
   - 0.0317147781317437 x3_input_acceleration + relaxation_CLF
   <= -20.07079283981647
 CBF_Constraint_obs2: 0.0459556738428372 x1_input_acceleration
   + 0.5064782712256413 x2_input_acceleration
   - 0.0619721588029487 x3_input_acceleration >= -0.2780908895609517
 CBF_Constraint_obs3: 0.4259556738428372 x1_input_acceleration
   + 0.5064782712256413 x2_input_acceleration
   - 0.0619721588029487 x3_input_acceleration >= -0.5022487797116471
 CBF_Constraint_obs1: - 0.2740443261571628 x1_input_acceleration
   + 0.5064782712256413 x2_input_acceleration
   - 0.0619721588029487 x3_input_acceleration >= -0.369326350486682
 CBF_Constraint_for_table: x3_input_acceleration >= -0.0840147056842787
 Relaxed_CLF_constraint: 0.084865883109773 x1_input_acceleration
   + 0.4090084732669343 x2_input_acceleration
   - 0.0302147781319431 x3_input_acceleration + relaxation_CLF
   <= -20.07016123285509
 CBF_Constraint_obs2: 0.0429556738430215 x1_input_acceleration
   + 0.5034782712256933 x2_input_acceleration
   - 0.0589721588032103 x3_input_acceleration >= -0.2735165937820399
 CBF_Constraint_obs3: 0.4229556738430215 x1_input_acceleration
   + 0.5034782712256933 x2_input_acceleration
   - 0.0589721588032103 x3_input_acceleration >= -0.4948244839329103
 CBF_Constraint_obs1: - 0.2770443261569785 x1_input_acceleration
   + 0.5034782712256933 x2_input_acceleration
   - 0.0589721588032103 x3_input_acceleration >= -0.3671520547076226
 CBF_Constraint_for_table: x3_input_acceleration >= -0.0915147056836246
 Relaxed_CLF_constraint: 0.0833658831098652 x1_input_acceleration
   + 0.4075084732669604 x2_input_acceleration
   - 0.0287147781320739 x3_input_acceleration + relaxation_CLF
   <= -20.06953502589369
 CBF_Constraint_obs2: 0.0399556738539716 x1_input_acceleration
   + 0.500478271228366 x2_input_acceleration
   - 0.0559721588161864 x3_input_acceleration >= -0.269009798009254
 CBF_Constraint_obs3: 0.4199556738539716 x1_input_acceleration
   + 0.500478271228366 x2_input_acceleration
   - 0.0559721588161864 x3_input_acceleration >= -0.487467688170527
 CBF_Constraint_obs1: - 0.2800443261460284 x1_input_acceleration
   + 0.500478271228366 x2_input_acceleration
   - 0.0559721588161864 x3_input_acceleration >= -0.3650452589260766
 CBF_Constraint_for_table: x3_input_acceleration >= -0.0990147056511844
 Relaxed_CLF_constraint: 0.0818658831153402 x1_input_acceleration
   + 0.4060084732682967 x2_input_acceleration
   - 0.0272147781385619 x3_input_acceleration + relaxation_CLF
   <= -20.06891421893322
 CBF_Constraint_obs2: 0.0369556738559301 x1_input_acceleration
   + 0.497478271228897 x2_input_acceleration
   - 0.0529721588232776 x3_input_acceleration >= -0.2645705022317955
 CBF_Constraint_obs3: 0.4169556738559301 x1_input_acceleration
   + 0.497478271228897 x2_input_acceleration
   - 0.0529721588232776 x3_input_acceleration >= -0.480178392394929
 CBF_Constraint_obs1: - 0.2830443261440699 x1_input_acceleration
   + 0.497478271228897 x2_input_acceleration
   - 0.0529721588232776 x3_input_acceleration >= -0.3630059631470514
 CBF_Constraint_for_table: x3_input_acceleration >= -0.1065147056334564
 Relaxed_CLF_constraint: 0.0803658831163195 x1_input_acceleration
   + 0.4045084732685622 x2_input_acceleration
   - 0.0257147781421075 x3_input_acceleration + relaxation_CLF
   <= -20.06829881197201
 CBF_Constraint_obs2: 0.0339556738798999 x1_input_acceleration
   + 0.4944782712351975 x2_input_acceleration
   - 0.0499721588689211 x3_input_acceleration >= -0.2601987064680105
 CBF_Constraint_obs3: 0.4139556738798999 x1_input_acceleration
   + 0.4944782712351975 x2_input_acceleration
   - 0.0499721588689211 x3_input_acceleration >= -0.4729565966539154
 CBF_Constraint_obs1: - 0.2860443261201001 x1_input_acceleration
   + 0.4944782712351975 x2_input_acceleration
   - 0.0499721588689211 x3_input_acceleration >= -0.3610341673640906
 CBF_Constraint_for_table: x3_input_acceleration >= -0.1140147055193475
 Relaxed_CLF_constraint: 0.0788658831283044 x1_input_acceleration
   + 0.4030084732717125 x2_input_acceleration
   - 0.0242147781649293 x3_input_acceleration + relaxation_CLF
   <= -20.06768880501279
 CBF_Constraint_obs2: 0.0309666411348882 x1_input_acceleration
   + 0.4914382169139982 x2_input_acceleration
   - 0.0469435486399768 x3_input_acceleration >= -0.2558426883321259
 CBF_Constraint_obs3: 0.4109556738802496 x1_input_acceleration
   + 0.4914782712352873 x2_input_acceleration
   - 0.046972158869469 x3_input_acceleration >= -0.465802300874569
 CBF_Constraint_obs1: - 0.2890443261197504 x1_input_acceleration
   + 0.4914782712352873 x2_input_acceleration
   - 0.046972158869469 x3_input_acceleration >= -0.3591298715841323
 CBF_Constraint_for_table: x3_input_acceleration >= -0.1218624986202423
 Relaxed_CLF_constraint: 0.0773658831284792 x1_input_acceleration
   + 0.4015084732717573 x2_input_acceleration
   - 0.0227147781652032 x3_input_acceleration + relaxation_CLF
   <= -20.06708419805134
 CBF_Constraint_obs2: 0.0279666411349674 x1_input_acceleration
   + 0.4884382169140151 x2_input_acceleration
   - 0.0439435486399796 x3_input_acceleration >= -0.251606325281986
 CBF_Constraint_obs3: 0.4079666411349674 x1_input_acceleration
   + 0.4884382169140151 x2_input_acceleration
   - 0.0439435486399796 x3_input_acceleration >= -0.4586746343602049
 CBF_Constraint_obs1: - 0.2920333588650326 x1_input_acceleration
   + 0.4884382169140151 x2_input_acceleration
   - 0.0439435486399796 x3_input_acceleration >= -0.3572330123740121
 CBF_Constraint_for_table: x3_input_acceleration >= -0.1293624986202352
 Relaxed_CLF_constraint: 0.0758658831285188 x1_input_acceleration
   + 0.4000084732717658 x2_input_acceleration
   - 0.0212147781652046 x3_input_acceleration + relaxation_CLF
   <= -20.06648499108987
 CBF_Constraint_obs2: 0.0249666411411972 x1_input_acceleration
   + 0.4854382169154999 x2_input_acceleration
   - 0.0409435486485523 x3_input_acceleration >= -0.2474374622348871
 CBF_Constraint_obs3: 0.4049666411411972 x1_input_acceleration
   + 0.4854382169154999 x2_input_acceleration
   - 0.0409435486485523 x3_input_acceleration >= -0.4516557713190245
 CBF_Constraint_obs1: - 0.2950333588588028 x1_input_acceleration
   + 0.4854382169154999 x2_input_acceleration
   - 0.0409435486485523 x3_input_acceleration >= -0.3554641493219293
 CBF_Constraint_for_table: x3_input_acceleration >= -0.1368624985988036
 Relaxed_CLF_constraint: 0.0743658831316337 x1_input_acceleration
   + 0.3985084732725082 x2_input_acceleration
   - 0.019714778169491 x3_input_acceleration + relaxation_CLF
   <= -20.06589118412888
 CBF_Constraint_obs2: 0.0219666411422927 x1_input_acceleration
   + 0.4824382169164164 x2_input_acceleration
   - 0.0379435486503981 x3_input_acceleration >= -0.2433360991859383
 CBF_Constraint_obs3: 0.4019666411422927 x1_input_acceleration
   + 0.4824382169164164 x2_input_acceleration
   - 0.0379435486503981 x3_input_acceleration >= -0.4447044082711165
 CBF_Constraint_obs1: - 0.2980333588577073 x1_input_acceleration
   + 0.4824382169164164 x2_input_acceleration
   - 0.0379435486503981 x3_input_acceleration >= -0.3537627862721042
 CBF_Constraint_for_table: x3_input_acceleration >= -0.1443624985941891
 Relaxed_CLF_constraint: 0.0728658831321815 x1_input_acceleration
   + 0.3970084732729665 x2_input_acceleration
   - 0.0182147781704139 x3_input_acceleration + relaxation_CLF
   <= -20.06530277716758
 CBF_Constraint_obs2: 0.0189666411428995 x1_input_acceleration
   + 0.4794382169166113 x2_input_acceleration
   - 0.0349435486508787 x3_input_acceleration >= -0.2393022361359244
 CBF_Constraint_obs3: 0.3989666411428995 x1_input_acceleration
   + 0.4794382169166113 x2_input_acceleration
   - 0.0349435486508787 x3_input_acceleration >= -0.4378205452216789
 CBF_Constraint_obs1: - 0.3010333588571005 x1_input_acceleration
   + 0.4794382169166113 x2_input_acceleration
   - 0.0349435486508787 x3_input_acceleration >= -0.3521289232216047
 CBF_Constraint_for_table: x3_input_acceleration >= -0.1518624985929876
 Relaxed_CLF_constraint: 0.0713658831324848 x1_input_acceleration
   + 0.3955084732730639 x2_input_acceleration
   - 0.0167147781706542 x3_input_acceleration + relaxation_CLF
   <= -20.06471977020615
 CBF_Constraint_obs2: 0.0159666411433949 x1_input_acceleration
   + 0.476438216916754 x2_input_acceleration
   - 0.0319435486519317 x3_input_acceleration >= -0.2353358730858703
 CBF_Constraint_obs3: 0.3959666411433949 x1_input_acceleration
   + 0.476438216916754 x2_input_acceleration
   - 0.0319435486519317 x3_input_acceleration >= -0.4310041821720955
 CBF_Constraint_obs1: - 0.3040333588566051 x1_input_acceleration
   + 0.476438216916754 x2_input_acceleration
   - 0.0319435486519317 x3_input_acceleration >= -0.3505625601711544
 CBF_Constraint_for_table: x3_input_acceleration >= -0.159362498590355
 Relaxed_CLF_constraint: 0.0698658831327325 x1_input_acceleration
   + 0.3940084732731353 x2_input_acceleration
   - 0.0152147781711807 x3_input_acceleration + relaxation_CLF
   <= -20.0641421632447
 CBF_Constraint_obs2: 0.0129437529616496 x1_input_acceleration
   + 0.4734857814242091 x2_input_acceleration
   - 0.0289809803747424 x3_input_acceleration >= -0.2314952789688841
 CBF_Constraint_obs3: 0.3929666411452434 x1_input_acceleration
   + 0.4734382169176783 x2_input_acceleration
   - 0.0289435486578235 x3_input_acceleration >= -0.4242553191250911
 CBF_Constraint_obs1: - 0.3070333588547566 x1_input_acceleration
   + 0.4734382169176783 x2_input_acceleration
   - 0.0289435486578235 x3_input_acceleration >= -0.3490636971209152
 CBF_Constraint_for_table: x3_input_acceleration >= -0.1668624985756256
 Relaxed_CLF_constraint: 0.0683658831336568 x1_input_acceleration
   + 0.3925084732735974 x2_input_acceleration
   - 0.0137147781741266 x3_input_acceleration + relaxation_CLF
   <= -20.06356995628343
 CBF_Constraint_obs2: 0.00994375296188621 x1_input_acceleration
   + 0.4704857814242711 x2_input_acceleration
   - 0.0259809803752148 x3_input_acceleration >= -0.2276634501082892
 CBF_Constraint_obs3: 0.3899437529618862 x1_input_acceleration
   + 0.4704857814242711 x2_input_acceleration
   - 0.0259809803752148 x3_input_acceleration >= -0.417610015422081
 CBF_Constraint_obs1: - 0.3100562470381138 x1_input_acceleration
   + 0.4704857814242711 x2_input_acceleration
   - 0.0259809803752148 x3_input_acceleration >= -0.3477084477387803
 CBF_Constraint_for_table: x3_input_acceleration >= -0.1743908107807129
 Relaxed_CLF_constraint: 0.0668658831337751 x1_input_acceleration
   + 0.3910084732736284 x2_input_acceleration
   - 0.0122147781743628 x3_input_acceleration + relaxation_CLF
   <= -20.06300314932195
 CBF_Constraint_obs2: 0.00694375296592048 x1_input_acceleration
   + 0.4674857814251093 x2_input_acceleration
   - 0.0229809803784911 x3_input_acceleration >= -0.2238991212488167
 CBF_Constraint_obs3: 0.3869437529659205 x1_input_acceleration
   + 0.4674857814251093 x2_input_acceleration
   - 0.0229809803784911 x3_input_acceleration >= -0.4109956865664411
 CBF_Constraint_obs1: - 0.3130562470340795 x1_input_acceleration
   + 0.4674857814251093 x2_input_acceleration
   - 0.0229809803784911 x3_input_acceleration >= -0.3463441188760803
 CBF_Constraint_for_table: x3_input_acceleration >= -0.1818908107725223
 Relaxed_CLF_constraint: 0.0653658831357922 x1_input_acceleration
   + 0.3895084732740475 x2_input_acceleration
   - 0.0107147781760009 x3_input_acceleration + relaxation_CLF
   <= -20.0624417423607
 CBF_Constraint_obs2: 0.0039437529677615 x1_input_acceleration
   + 0.4644857814255787 x2_input_acceleration
   - 0.0199809803824787 x3_input_acceleration >= -0.2202022923888077
 CBF_Constraint_obs3: 0.3839437529677615 x1_input_acceleration
   + 0.4644857814255787 x2_input_acceleration
   - 0.0199809803824787 x3_input_acceleration >= -0.4044488577081812
 CBF_Constraint_obs1: - 0.3160562470322385 x1_input_acceleration
   + 0.4644857814255787 x2_input_acceleration
   - 0.0199809803824787 x3_input_acceleration >= -0.3450472900145986
 CBF_Constraint_for_table: x3_input_acceleration >= -0.1889023506986189
 Relaxed_CLF_constraint: 0.0638658831367127 x1_input_acceleration
   + 0.3880084732742822 x2_input_acceleration
   - 0.0092147781779947 x3_input_acceleration + relaxation_CLF
   <= -20.06188573539934
 CBF_Constraint_obs2: 8.7175058133914796e-04 x1_input_acceleration
   + 0.4611324451034181 x2_input_acceleration
   - 0.0172588572686246 x3_input_acceleration >= -0.2161772000379397
 CBF_Constraint_obs3: 0.3808717505813392 x1_input_acceleration
   + 0.4611324451034181 x2_input_acceleration
   - 0.0172588572686246 x3_input_acceleration >= -0.3975053630902119
 CBF_Constraint_obs1: - 0.3190562470077722 x1_input_acceleration
   + 0.4614857814376466 x2_input_acceleration
   - 0.0169809804146817 x3_input_acceleration >= -0.3438179611495545
 CBF_Constraint_for_table: x3_input_acceleration >= -0.1964023506181115
 Relaxed_CLF_constraint: 0.0623658831489459 x1_input_acceleration
   + 0.3865084732803162 x2_input_acceleration
   - 0.00771477819409619 x3_input_acceleration + relaxation_CLF
   <= -20.0613351284404
 CBF_Constraint_obs2: - 0.00212824941862561 x1_input_acceleration
   + 0.4581324451034398 x2_input_acceleration
   - 0.0142588572686652 x3_input_acceleration >= -0.2126164771408154
 CBF_Constraint_obs3: 0.3778717505813744 x1_input_acceleration
   + 0.4581324451034398 x2_input_acceleration
   - 0.0142588572686652 x3_input_acceleration >= -0.3910946401931212
 CBF_Constraint_obs1: - 0.3221282494186256 x1_input_acceleration
   + 0.4581324451034398 x2_input_acceleration
   - 0.0142588572686652 x3_input_acceleration >= -0.3423190766757159
 CBF_Constraint_for_table: x3_input_acceleration >= -0.2039023506180097
 Relaxed_CLF_constraint: 0.0608658831489635 x1_input_acceleration
   + 0.3850084732803271 x2_input_acceleration
   - 0.00621477819411653 x3_input_acceleration + relaxation_CLF
   <= -20.06078992147886
 CBF_Constraint_obs2: - 0.00512824941372386 x1_input_acceleration
   + 0.4551324451047563 x2_input_acceleration
   - 0.0112588572727996 x3_input_acceleration >= -0.2091232542452158
 CBF_Constraint_obs3: 0.3748717505862761 x1_input_acceleration
   + 0.4551324451047563 x2_input_acceleration
   - 0.0112588572727996 x3_input_acceleration >= -0.3847514173021781
 CBF_Constraint_obs1: - 0.3251282494137239 x1_input_acceleration
   + 0.4551324451047563 x2_input_acceleration
   - 0.0112588572727996 x3_input_acceleration >= -0.3412258537761949
 CBF_Constraint_for_table: x3_input_acceleration >= -0.2114023506076738
 Relaxed_CLF_constraint: 0.0593658831514144 x1_input_acceleration
   + 0.3835084732809853 x2_input_acceleration
   - 0.00471477819618372 x3_input_acceleration + relaxation_CLF
   <= -20.06025011451764
 CBF_Constraint_obs2: - 0.00812824941371149 x1_input_acceleration
   + 0.4521324451047619 x2_input_acceleration
   - 0.00825885727281736 x3_input_acceleration >= -0.2056975313479934
 CBF_Constraint_obs3: 0.3718717505862885 x1_input_acceleration
   + 0.4521324451047619 x2_input_acceleration
   - 0.00825885727281736 x3_input_acceleration >= -0.3784756944049675
 CBF_Constraint_obs1: - 0.3281282494137115 x1_input_acceleration
   + 0.4521324451047619 x2_input_acceleration
   - 0.00825885727281736 x3_input_acceleration >= -0.3402001308789626
 CBF_Constraint_for_table: x3_input_acceleration >= -0.2189023506076294
 Relaxed_CLF_constraint: 0.0578658831514205 x1_input_acceleration
   + 0.3820084732809881 x2_input_acceleration
   - 0.00321477819619259 x3_input_acceleration + relaxation_CLF
   <= -20.05971570755609
 CBF_Constraint_obs2: - 0.011128249412767 x1_input_acceleration
   + 0.4491324451050189 x2_input_acceleration
   - 0.00525885727468706 x3_input_acceleration >= -0.2023393084510513
 CBF_Constraint_obs3: 0.368871750587233 x1_input_acceleration
   + 0.4491324451050189 x2_input_acceleration
   - 0.00525885727468706 x3_input_acceleration >= -0.3722674715089226
 CBF_Constraint_obs1: - 0.331128249412767 x1_input_acceleration
   + 0.4491324451050189 x2_input_acceleration
   - 0.00525885727468706 x3_input_acceleration >= -0.3392419079812649
 CBF_Constraint_for_table: x3_input_acceleration >= -0.2264023506029552
 Relaxed_CLF_constraint: 0.0563658831518928 x1_input_acceleration
   + 0.3805084732811166 x2_input_acceleration
   - 0.00171477819712744 x3_input_acceleration + relaxation_CLF
   <= -20.05918670059459
 CBF_Constraint_obs2: - 0.0141253883814165 x1_input_acceleration
   + 0.4461270806901951 x2_input_acceleration
   - 0.00225480417940585 x3_input_acceleration >= -0.1990424785799614
 CBF_Constraint_obs3: 0.3658717505956343 x1_input_acceleration
   + 0.4461324451082249 x2_input_acceleration
   - 0.00225885729525058 x3_input_acceleration >= -0.3661267486230469
 CBF_Constraint_obs1: - 0.3341282494043658 x1_input_acceleration
   + 0.4461324451082249 x2_input_acceleration
   - 0.00225885729525058 x3_input_acceleration >= -0.3383511850806871
 CBF_Constraint_for_table: x3_input_acceleration >= -0.2343464051551231
 Relaxed_CLF_constraint: 0.0548658831560934 x1_input_acceleration
   + 0.3790084732827196 x2_input_acceleration
   - 2.1477820740919973e-04 x3_input_acceleration + relaxation_CLF
   <= -20.05866309363371
 CBF_Constraint_obs2: - 0.0171253883495844 x1_input_acceleration
   + 0.4431270806992999 x2_input_acceleration
   + 7.4519580649284878e-04 x3_input_acceleration >= -0.1958193048649974
 CBF_Constraint_obs3: 0.3628746116504156 x1_input_acceleration
   + 0.4431270806992999 x2_input_acceleration
   + 7.4519580649284878e-04 x3_input_acceleration >= -0.3600501859328921
 CBF_Constraint_obs1: - 0.3371253883495844 x1_input_acceleration
   + 0.4431270806992999 x2_input_acceleration
   + 7.4519580649284878e-04 x3_input_acceleration >= -0.337519615544665
 CBF_Constraint_for_table: x3_input_acceleration >= -0.2418464051198698
 Relaxed_CLF_constraint: 0.0533658831720095 x1_input_acceleration
   + 0.377508473287272 x2_input_acceleration
   + 0.00128522178554015 x3_input_acceleration + relaxation_CLF
   <= -20.05814488667418
 CBF_Constraint_obs2: - 0.0201253883470561 x1_input_acceleration
   + 0.4401270807001876 x2_input_acceleration
   + 0.00374519579907415 x3_input_acceleration >= -0.1926636311417033
 CBF_Constraint_obs3: 0.3598746116529439 x1_input_acceleration
   + 0.4401270807001876 x2_input_acceleration
   + 0.00374519579907415 x3_input_acceleration >= -0.3540445122120001
 CBF_Constraint_obs1: - 0.3401253883470561 x1_input_acceleration
   + 0.4401270807001876 x2_input_acceleration
   + 0.00374519579907415 x3_input_acceleration >= -0.3367639418193481
 CBF_Constraint_for_table: x3_input_acceleration >= -0.2493464051013231
 Relaxed_CLF_constraint: 0.0518658831732736 x1_input_acceleration
   + 0.3760084732877158 x2_input_acceleration
   + 0.0027852217818308 x3_input_acceleration + relaxation_CLF
   <= -20.05763207971275
 CBF_Constraint_obs2: - 0.0231253883437248 x1_input_acceleration
   + 0.4371270807010135 x2_input_acceleration
   + 0.00674519578415475 x3_input_acceleration >= -0.1895754574180062
 CBF_Constraint_obs3: 0.3568746116562752 x1_input_acceleration
   + 0.4371270807010135 x2_input_acceleration
   + 0.00674519578415475 x3_input_acceleration >= -0.3481063384914676
 CBF_Constraint_obs1: - 0.3431253883437248 x1_input_acceleration
   + 0.4371270807010135 x2_input_acceleration
   + 0.00674519578415475 x3_input_acceleration >= -0.336075768092986
 CBF_Constraint_for_table: x3_input_acceleration >= -0.2568464050640246
 Relaxed_CLF_constraint: 0.0503658831749393 x1_input_acceleration
   + 0.3745084732881288 x2_input_acceleration
   + 0.0042852217743711 x3_input_acceleration + relaxation_CLF
   <= -20.0571246727513
 CBF_Constraint_obs2: - 0.0261253883413101 x1_input_acceleration
   + 0.4341270807018867 x2_input_acceleration
   + 0.00974519577585928 x3_input_acceleration >= -0.1865547836942957
 CBF_Constraint_obs3: 0.3538746116586899 x1_input_acceleration
   + 0.4341270807018867 x2_input_acceleration
   + 0.00974519577585928 x3_input_acceleration >= -0.3422356647700511
 CBF_Constraint_obs1: - 0.3461253883413101 x1_input_acceleration
   + 0.4341270807018867 x2_input_acceleration
   + 0.00974519577585928 x3_input_acceleration >= -0.3354550943673437
 CBF_Constraint_for_table: x3_input_acceleration >= -0.2643464050432859
 Relaxed_CLF_constraint: 0.0488658831761467 x1_input_acceleration
   + 0.3730084732885653 x2_input_acceleration
   + 0.00578522177022336 x3_input_acceleration + relaxation_CLF
   <= -20.05662266578983
 CBF_Constraint_obs2: - 0.0291253883323406 x1_input_acceleration
   + 0.4311270807037713 x2_input_acceleration
   + 0.0127451957659433 x3_input_acceleration >= -0.1836016099709725
 CBF_Constraint_obs3: 0.3508746116676594 x1_input_acceleration
   + 0.4311270807037713 x2_input_acceleration
   + 0.0127451957659433 x3_input_acceleration >= -0.336432491055249
 CBF_Constraint_obs1: - 0.3491253883323406 x1_input_acceleration
   + 0.4311270807037713 x2_input_acceleration
   + 0.0127451957659433 x3_input_acceleration >= -0.3349019206368449
 CBF_Constraint_for_table: x3_input_acceleration >= -0.2718464050184959
 Relaxed_CLF_constraint: 0.0473658831806314 x1_input_acceleration
   + 0.3715084732895076 x2_input_acceleration
   + 0.00728522176526536 x3_input_acceleration + relaxation_CLF
   <= -20.05612605882862
 CBF_Constraint_obs2: - 0.0321249114891604 x1_input_acceleration
   + 0.4281322067049216 x2_input_acceleration
   + 0.0157530635684935 x3_input_acceleration >= -0.1807216942100064
 CBF_Constraint_obs3: 0.3478746116736814 x1_input_acceleration
   + 0.428127080705471 x2_input_acceleration
   + 0.0157451957553831 x3_input_acceleration >= -0.3306968173373486
 CBF_Constraint_obs1: - 0.3521253883263186 x1_input_acceleration
   + 0.428127080705471 x2_input_acceleration
   + 0.0157451957553831 x3_input_acceleration >= -0.3344162469084061
 CBF_Constraint_for_table: x3_input_acceleration >= -0.2794101819620051
 Relaxed_CLF_constraint: 0.0458658831836424 x1_input_acceleration
   + 0.3700084732903575 x2_input_acceleration
   + 0.00878522175998528 x3_input_acceleration + relaxation_CLF
   <= -20.0556348518673
 CBF_Constraint_obs2: - 0.035124911489146 x1_input_acceleration
   + 0.4251322067049245 x2_input_acceleration
   + 0.018753063568483 x3_input_acceleration >= -0.1779035374726532
 CBF_Constraint_obs3: 0.344875088510854 x1_input_acceleration
   + 0.4251322067049245 x2_input_acceleration
   + 0.018753063568483 x3_input_acceleration >= -0.3250348715579646
 CBF_Constraint_obs1: - 0.355124911489146 x1_input_acceleration
   + 0.4251322067049245 x2_input_acceleration
   + 0.018753063568483 x3_input_acceleration >= -0.3340034666639701
 CBF_Constraint_for_table: x3_input_acceleration >= -0.2869101819619789
 Relaxed_CLF_constraint: 0.0443658831836496 x1_input_acceleration
   + 0.368508473290359 x2_input_acceleration
   + 0.01028522175998 x3_input_acceleration + relaxation_CLF
   <= -20.05514904490564
 CBF_Constraint_obs2: - 0.0379804298301538 x1_input_acceleration
   + 0.422529054429871 x2_input_acceleration
   + 0.0220450071183114 x3_input_acceleration >= -0.1755741215329132
 CBF_Constraint_obs3: 0.3420195701698462 x1_input_acceleration
   + 0.422529054429871 x2_input_acceleration
   + 0.0220450071183114 x3_input_acceleration >= -0.319992713194267
 CBF_Constraint_obs1: - 0.3579804298301538 x1_input_acceleration
   + 0.422529054429871 x2_input_acceleration
   + 0.0220450071183114 x3_input_acceleration >= -0.3339584653970362
 CBF_Constraint_for_table: x3_input_acceleration >= -0.2943171987154258
 Relaxed_CLF_constraint: 0.0428658831836779 x1_input_acceleration
   + 0.3670084732903749 x2_input_acceleration
   + 0.0117852217598393 x3_input_acceleration + relaxation_CLF
   <= -20.05466863794399
 CBF_Constraint_obs2: - 0.0409804298300435 x1_input_acceleration
   + 0.4195290544299082 x2_input_acceleration
   + 0.0250450071179838 x3_input_acceleration >= -0.1728890944018098
 CBF_Constraint_obs3: 0.3390195701699565 x1_input_acceleration
   + 0.4195290544299082 x2_input_acceleration
   + 0.0250450071179838 x3_input_acceleration >= -0.3144576860632685
 CBF_Constraint_obs1: - 0.3609804298300435 x1_input_acceleration
   + 0.4195290544299082 x2_input_acceleration
   + 0.0250450071179838 x3_input_acceleration >= -0.3336734382658446
 CBF_Constraint_for_table: x3_input_acceleration >= -0.3018171987146068
 Relaxed_CLF_constraint: 0.0413658831837331 x1_input_acceleration
   + 0.3655084732903935 x2_input_acceleration
   + 0.0132852217596755 x3_input_acceleration + relaxation_CLF
   <= -20.05419363098234
 CBF_Constraint_obs2: - 0.0439804298299991 x1_input_acceleration
   + 0.41652905442992 x2_input_acceleration
   + 0.0280450071179072 x3_input_acceleration >= -0.1702715672706977
 CBF_Constraint_obs3: 0.3360195701700009 x1_input_acceleration
   + 0.41652905442992 x2_input_acceleration
   + 0.0280450071179072 x3_input_acceleration >= -0.3089901589321986
 CBF_Constraint_obs1: - 0.3639804298299991 x1_input_acceleration
   + 0.41652905442992 x2_input_acceleration
   + 0.0280450071179072 x3_input_acceleration >= -0.333455911134697
 CBF_Constraint_for_table: x3_input_acceleration >= -0.3093171987144152
 Relaxed_CLF_constraint: 0.0398658831837553 x1_input_acceleration
   + 0.3640084732903994 x2_input_acceleration
   + 0.0147852217596372 x3_input_acceleration + relaxation_CLF
   <= -20.05372402402068
 CBF_Constraint_obs2: - 0.0469804298296747 x1_input_acceleration
   + 0.4135290544300076 x2_input_acceleration
   + 0.0310450071171323 x3_input_acceleration >= -0.167721540139575
 CBF_Constraint_obs3: 0.3330195701703254 x1_input_acceleration
   + 0.4135290544300076 x2_input_acceleration
   + 0.0310450071171323 x3_input_acceleration >= -0.3035901318013841
 CBF_Constraint_obs1: - 0.3669804298296747 x1_input_acceleration
   + 0.4135290544300076 x2_input_acceleration
   + 0.0310450071171323 x3_input_acceleration >= -0.3333058840033147
 CBF_Constraint_for_table: x3_input_acceleration >= -0.316817198712478
 Relaxed_CLF_constraint: 0.0383658831839175 x1_input_acceleration
   + 0.3625084732904432 x2_input_acceleration
   + 0.0162852217592498 x3_input_acceleration + relaxation_CLF
   <= -20.05325981705904
 CBF_Constraint_obs2: - 0.0499804298241833 x1_input_acceleration
   + 0.4105290544317157 x2_input_acceleration
   + 0.0340450071112384 x3_input_acceleration >= -0.1652390130090162
 CBF_Constraint_obs3: 0.3300195701758167 x1_input_acceleration
   + 0.4105290544317157 x2_input_acceleration
   + 0.0340450071112384 x3_input_acceleration >= -0.2982576046760421
 CBF_Constraint_obs1: - 0.3699804298241833 x1_input_acceleration
   + 0.4105290544317157 x2_input_acceleration
   + 0.0340450071112384 x3_input_acceleration >= -0.3332233568683628
 CBF_Constraint_for_table: x3_input_acceleration >= -0.3243171986977433
 Relaxed_CLF_constraint: 0.0368658831866632 x1_input_acceleration
   + 0.3610084732912973 x2_input_acceleration
   + 0.0177852217563028 x3_input_acceleration + relaxation_CLF
   <= -20.05280101009767
 CBF_Constraint_obs2: - 0.0530080863734375 x1_input_acceleration
   + 0.4071236236397098 x2_input_acceleration
   + 0.0368385366152288 x3_input_acceleration >= -0.1623957249080668
 CBF_Constraint_obs3: 0.3270195701817383 x1_input_acceleration
   + 0.407529054433472 x2_input_acceleration
   + 0.0370450071047308 x3_input_acceleration >= -0.2929925775508478
 CBF_Constraint_obs1: - 0.3729804298182617 x1_input_acceleration
   + 0.407529054433472 x2_input_acceleration
   + 0.0370450071047308 x3_input_acceleration >= -0.3332083297328058
 CBF_Constraint_for_table: x3_input_acceleration >= -0.3318171986814742
 Relaxed_CLF_constraint: 0.035365883189624 x1_input_acceleration
   + 0.3595084732921754 x2_input_acceleration
   + 0.019285221753049 x3_input_acceleration + relaxation_CLF
   <= -20.05234760313629
 CBF_Constraint_obs2: - 0.0560080863668507 x1_input_acceleration
   + 0.4041236236415635 x2_input_acceleration
   + 0.0398385366059576 x3_input_acceleration >= -0.1600498974032111
 CBF_Constraint_obs3: 0.3240195701883251 x1_input_acceleration
   + 0.4045290544353257 x2_input_acceleration
   + 0.0400450070954595 x3_input_acceleration >= -0.2877950504258013
 CBF_Constraint_obs1: - 0.3759804298116749 x1_input_acceleration
   + 0.4045290544353257 x2_input_acceleration
   + 0.0400450070954595 x3_input_acceleration >= -0.3332608025962324
 CBF_Constraint_for_table: x3_input_acceleration >= -0.338958676719972
 Relaxed_CLF_constraint: 0.0338658831929174 x1_input_acceleration
   + 0.3580084732931023 x2_input_acceleration
   + 0.0207852217484134 x3_input_acceleration + relaxation_CLF
   <= -20.05189959617489
 CBF_Constraint_obs2: - 0.059008086362064 x1_input_acceleration
   + 0.4011236236430838 x2_input_acceleration
   + 0.042838536600909 x3_input_acceleration >= -0.1577715698984733
 CBF_Constraint_obs3: 0.320991913637936 x1_input_acceleration
   + 0.4011236236430838 x2_input_acceleration
   + 0.042838536600909 x3_input_acceleration >= -0.2822138878545124
 CBF_Constraint_obs1: - 0.379008086362064 x1_input_acceleration
   + 0.4011236236430838 x2_input_acceleration
   + 0.042838536600909 x3_input_acceleration >= -0.3329780389881244
 CBF_Constraint_for_table: x3_input_acceleration >= -0.3464586767073507
 Relaxed_CLF_constraint: 0.0323658831953107 x1_input_acceleration
   + 0.3565084732938624 x2_input_acceleration
   + 0.0222852217458891 x3_input_acceleration + relaxation_CLF
   <= -20.05145698921344
 CBF_Constraint_obs2: - 0.0620080863577065 x1_input_acceleration
   + 0.3981236236444116 x2_input_acceleration
   + 0.045838536595508 x3_input_acceleration >= -0.1555607423933995
 CBF_Constraint_obs3: 0.3179919136422935 x1_input_acceleration
   + 0.3981236236444116 x2_input_acceleration
   + 0.045838536595508 x3_input_acceleration >= -0.2771530603535783
 CBF_Constraint_obs1: - 0.3820080863577065 x1_input_acceleration
   + 0.3981236236444116 x2_input_acceleration
   + 0.045838536595508 x3_input_acceleration >= -0.3331672114795647
 CBF_Constraint_for_table: x3_input_acceleration >= -0.353958676693848
 Relaxed_CLF_constraint: 0.0308658831974895 x1_input_acceleration
   + 0.3550084732945263 x2_input_acceleration
   + 0.0237852217431886 x3_input_acceleration + relaxation_CLF
   <= -20.05101978225194
 CBF_Constraint_obs2: - 0.0650080863554708 x1_input_acceleration
   + 0.3951236236452084 x2_input_acceleration
   + 0.0488385365919332 x3_input_acceleration >= -0.1534174148882028
 CBF_Constraint_obs3: 0.3149919136445292 x1_input_acceleration
   + 0.3951236236452084 x2_input_acceleration
   + 0.0488385365919332 x3_input_acceleration >= -0.2721597328505055
 CBF_Constraint_obs1: - 0.3850080863554708 x1_input_acceleration
   + 0.3951236236452084 x2_input_acceleration
   + 0.0488385365919332 x3_input_acceleration >= -0.3334238839725795
 CBF_Constraint_for_table: x3_input_acceleration >= -0.3614586766849111
 Relaxed_CLF_constraint: 0.0293658831986073 x1_input_acceleration
   + 0.3535084732949247 x2_input_acceleration
   + 0.0252852217414012 x3_input_acceleration + relaxation_CLF
   <= -20.05058797529034
 CBF_Constraint_obs2: - 0.068031451373372 x1_input_acceleration
   + 0.3924860198862339 x2_input_acceleration
   + 0.052023668613164 x3_input_acceleration >= -0.1517250203475962
 CBF_Constraint_obs3: 0.3119919136473799 x1_input_acceleration
   + 0.3921236236459995 x2_input_acceleration
   + 0.0518385365864916 x3_input_acceleration >= -0.2672339053475658
 CBF_Constraint_obs1: - 0.3880080863526201 x1_input_acceleration
   + 0.3921236236459995 x2_input_acceleration
   + 0.0518385365864916 x3_input_acceleration >= -0.333748056464651
 CBF_Constraint_for_table: x3_input_acceleration >= -0.3689071186535764
 Relaxed_CLF_constraint: 0.0278658832000327 x1_input_acceleration
   + 0.3520084732953203 x2_input_acceleration
   + 0.0267852217386804 x3_input_acceleration + relaxation_CLF
   <= -20.05016156832873
 CBF_Constraint_obs2: - 0.0710314513702624 x1_input_acceleration
   + 0.3894860198869863 x2_input_acceleration
   + 0.0550236686116124 x3_input_acceleration >= -0.1497155385983154
 CBF_Constraint_obs3: 0.3089685486297376 x1_input_acceleration
   + 0.3894860198869863 x2_input_acceleration
   + 0.0550236686116124 x3_input_acceleration >= -0.2627356597965662
 CBF_Constraint_obs1: - 0.3910314513702624 x1_input_acceleration
   + 0.3894860198869863 x2_input_acceleration
   + 0.0550236686116124 x3_input_acceleration >= -0.3345406996945254
 CBF_Constraint_for_table: x3_input_acceleration >= -0.3764071186496975
 Relaxed_CLF_constraint: 0.0263658832015875 x1_input_acceleration
   + 0.3505084732956965 x2_input_acceleration
   + 0.0282852217379046 x3_input_acceleration + relaxation_CLF
   <= -20.04974056136714
 CBF_Constraint_obs2: - 0.0740314513637528 x1_input_acceleration
   + 0.3864860198885638 x2_input_acceleration
   + 0.0580236686064652 x3_input_acceleration >= -0.1477735568485999
 CBF_Constraint_obs3: 0.3059685486362472 x1_input_acceleration
   + 0.3864860198885638 x2_input_acceleration
   + 0.0580236686064652 x3_input_acceleration >= -0.2579436780530348
 CBF_Constraint_obs1: - 0.3940314513637528 x1_input_acceleration
   + 0.3864860198885638 x2_input_acceleration
   + 0.0580236686064652 x3_input_acceleration >= -0.3349987179396022
 CBF_Constraint_for_table: x3_input_acceleration >= -0.3839071186368296
 Relaxed_CLF_constraint: 0.0248658832048423 x1_input_acceleration
   + 0.3490084732964852 x2_input_acceleration
   + 0.029785221735331 x3_input_acceleration + relaxation_CLF
   <= -20.04932495440565
 CBF_Constraint_obs2: - 0.0770548163522998 x1_input_acceleration
   + 0.3834799402271095 x2_input_acceleration
   + 0.0610291521331182 x3_input_acceleration >= -0.1458985833622017
 CBF_Constraint_obs3: 0.3029451836477002 x1_input_acceleration
   + 0.3834799402271095 x2_input_acceleration
   + 0.0610291521331182 x3_input_acceleration >= -0.253196507827517
 CBF_Constraint_obs1: - 0.3970548163522998 x1_input_acceleration
   + 0.3834799402271095 x2_input_acceleration
   + 0.0610291521331182 x3_input_acceleration >= -0.3355424364440415
 CBF_Constraint_for_table: x3_input_acceleration >= -0.3915188770941175
 Relaxed_CLF_constraint: 0.0233658832209448 x1_input_acceleration
   + 0.3475084733026416 x2_input_acceleration
   + 0.0312852216849978 x3_input_acceleration + relaxation_CLF
   <= -20.04891474744468
 CBF_Constraint_obs2: - 0.0800548163509657 x1_input_acceleration
   + 0.3804799402274451 x2_input_acceleration
   + 0.0640291521299439 x3_input_acceleration >= -0.1440918635736832
 CBF_Constraint_obs3: 0.2999451836490343 x1_input_acceleration
   + 0.3804799402274451 x2_input_acceleration
   + 0.0640291521299439 x3_input_acceleration >= -0.2485397880402658
 CBF_Constraint_obs1: - 0.4000548163509657 x1_input_acceleration
   + 0.3804799402274451 x2_input_acceleration
   + 0.0640291521299439 x3_input_acceleration >= -0.3361357166544559
 CBF_Constraint_for_table: x3_input_acceleration >= -0.3990188770861818
 Relaxed_CLF_constraint: 0.0218658832216118 x1_input_acceleration
   + 0.3460084733028094 x2_input_acceleration
   + 0.0327852216834107 x3_input_acceleration + relaxation_CLF
   <= -20.04850994048289
 CBF_Constraint_obs2: - 0.0830548163502748 x1_input_acceleration
   + 0.3774799402276461 x2_input_acceleration
   + 0.0670291521288879 x3_input_acceleration >= -0.1423526437854535
 CBF_Constraint_obs3: 0.2969451836497252 x1_input_acceleration
   + 0.3774799402276461 x2_input_acceleration
   + 0.0670291521288879 x3_input_acceleration >= -0.2439505682526924
 CBF_Constraint_obs1: - 0.4030548163502748 x1_input_acceleration
   + 0.3774799402276461 x2_input_acceleration
   + 0.0670291521288879 x3_input_acceleration >= -0.3367964968656734
 CBF_Constraint_for_table: x3_input_acceleration >= -0.4065188770835417
 Relaxed_CLF_constraint: 0.0203658832219573 x1_input_acceleration
   + 0.3445084733029099 x2_input_acceleration
   + 0.0342852216828827 x3_input_acceleration + relaxation_CLF
   <= -20.0481105335211
 CBF_Constraint_obs2: - 0.0860548163493063 x1_input_acceleration
   + 0.3744799402278788 x2_input_acceleration
   + 0.0700291521276658 x3_input_acceleration >= -0.1406809239971353
 CBF_Constraint_obs3: 0.2939451836506937 x1_input_acceleration
   + 0.3744799402278788 x2_input_acceleration
   + 0.0700291521276658 x3_input_acceleration >= -0.2394288484652943
 CBF_Constraint_obs1: - 0.4060548163493063 x1_input_acceleration
   + 0.3744799402278788 x2_input_acceleration
   + 0.0700291521276658 x3_input_acceleration >= -0.3375247770765805
 CBF_Constraint_for_table: x3_input_acceleration >= -0.4140188770804864
 Relaxed_CLF_constraint: 0.0188658832224415 x1_input_acceleration
   + 0.3430084733030263 x2_input_acceleration
   + 0.0357852216822716 x3_input_acceleration + relaxation_CLF
   <= -20.04771652655931
 CBF_Constraint_obs2: - 0.089128249268584 x1_input_acceleration
   + 0.3711329219896736 x2_input_acceleration
   + 0.0727448379449176 x3_input_acceleration >= -0.1387191275623854
 CBF_Constraint_obs3: 0.2909451836537792 x1_input_acceleration
   + 0.3714799402315559 x2_input_acceleration
   + 0.0730291521004962 x3_input_acceleration >= -0.2349746286778613
 CBF_Constraint_obs1: - 0.4090548163462208 x1_input_acceleration
   + 0.3714799402315559 x2_input_acceleration
   + 0.0730291521004962 x3_input_acceleration >= -0.3383205572837477
 CBF_Constraint_for_table: x3_input_acceleration >= -0.4214649348090407
 Relaxed_CLF_constraint: 0.0173658832239843 x1_input_acceleration
   + 0.3415084733048648 x2_input_acceleration
   + 0.0372852216686868 x3_input_acceleration + relaxation_CLF
   <= -20.04732791959762
 CBF_Constraint_obs2: - 0.0921282492672429 x1_input_acceleration
   + 0.3681329219900388 x2_input_acceleration
   + 0.0757448379310721 x3_input_acceleration >= -0.1371834287989696
 CBF_Constraint_obs3: 0.2878717507327571 x1_input_acceleration
   + 0.3681329219900388 x2_input_acceleration
   + 0.0757448379310721 x3_input_acceleration >= -0.2301615919950887
 CBF_Constraint_obs1: - 0.4120548163448797 x1_input_acceleration
   + 0.3684799402319212 x2_input_acceleration
   + 0.0760291520866507 x3_input_acceleration >= -0.3391838374916847
 CBF_Constraint_for_table: x3_input_acceleration >= -0.4289649347744271
 Relaxed_CLF_constraint: 0.0158658832246548 x1_input_acceleration
   + 0.3400084733050475 x2_input_acceleration
   + 0.0387852216617641 x3_input_acceleration + relaxation_CLF
   <= -20.04694471263563
 CBF_Constraint_obs2: - 0.0951282492655148 x1_input_acceleration
   + 0.3651329219904379 x2_input_acceleration
   + 0.0787448379119643 x3_input_acceleration >= -0.1357152300342233
 CBF_Constraint_obs3: 0.2848717507344852 x1_input_acceleration
   + 0.3651329219904379 x2_input_acceleration
   + 0.0787448379119643 x3_input_acceleration >= -0.2258433932319842
 CBF_Constraint_obs1: - 0.4151282492655148 x1_input_acceleration
   + 0.3651329219904379 x2_input_acceleration
   + 0.0787448379119643 x3_input_acceleration >= -0.3398178294466352
 CBF_Constraint_for_table: x3_input_acceleration >= -0.4364649347266575
 Relaxed_CLF_constraint: 0.0143658832255189 x1_input_acceleration
   + 0.338508473305247 x2_input_acceleration
   + 0.0402852216522102 x3_input_acceleration + relaxation_CLF
   <= -20.04656690567355
 CBF_Constraint_obs2: - 0.0981282492655032 x1_input_acceleration
   + 0.3621329219904399 x2_input_acceleration
   + 0.081744837911938 x3_input_acceleration >= -0.1343145312731197
 CBF_Constraint_obs3: 0.2818717507344968 x1_input_acceleration
   + 0.3621329219904399 x2_input_acceleration
   + 0.081744837911938 x3_input_acceleration >= -0.2215926944708917
 CBF_Constraint_obs1: - 0.4181282492655032 x1_input_acceleration
   + 0.3621329219904399 x2_input_acceleration
   + 0.081744837911938 x3_input_acceleration >= -0.3408171306855223
 CBF_Constraint_for_table: x3_input_acceleration >= -0.4439649347265919
 Relaxed_CLF_constraint: 0.0128658832255247 x1_input_acceleration
   + 0.337008473305248 x2_input_acceleration
   + 0.0417852216521971 x3_input_acceleration + relaxation_CLF
   <= -20.04619449871169
 CBF_Constraint_obs2: - 0.1011282492641207 x1_input_acceleration
   + 0.3591329219910503 x2_input_acceleration
   + 0.0847448379100703 x3_input_acceleration >= -0.132981332511825
 CBF_Constraint_obs3: 0.2788717507358793 x1_input_acceleration
   + 0.3591329219910503 x2_input_acceleration
   + 0.0847448379100703 x3_input_acceleration >= -0.2174094957109104
 CBF_Constraint_obs1: - 0.4211282492641207 x1_input_acceleration
   + 0.3591329219910503 x2_input_acceleration
   + 0.0847448379100703 x3_input_acceleration >= -0.3418839319231215
 CBF_Constraint_for_table: x3_input_acceleration >= -0.4514649347219225
 Relaxed_CLF_constraint: 0.011365883226216 x1_input_acceleration
   + 0.3355084733055532 x2_input_acceleration
   + 0.0432852216512632 x3_input_acceleration + relaxation_CLF
   <= -20.04582749174989
 CBF_Constraint_obs2: - 0.1039813833907906 x1_input_acceleration
   + 0.3565303657714897 x2_input_acceleration
   + 0.0880811272449944 x3_input_acceleration >= -0.1321053934822046
 CBF_Constraint_obs3: 0.2758717507644828 x1_input_acceleration
   + 0.3561329220001274 x2_input_acceleration
   + 0.0877448378391716 x3_input_acceleration >= -0.2132937969620403
 CBF_Constraint_obs1: - 0.4241282492355172 x1_input_acceleration
   + 0.3561329220001274 x2_input_acceleration
   + 0.0877448378391716 x3_input_acceleration >= -0.3430182331241954
 CBF_Constraint_for_table: x3_input_acceleration >= -0.4589649345446759
 Relaxed_CLF_constraint: 0.00986588324051771 x1_input_acceleration
   + 0.3340084733100917 x2_input_acceleration
   + 0.0447852216158139 x3_input_acceleration + relaxation_CLF
   <= -20.04546588478809
 CBF_Constraint_obs2: - 0.1071373091409354 x1_input_acceleration
   + 0.3531268423265124 x2_input_acceleration
   + 0.090735062676721 x3_input_acceleration >= -0.1305122767255029
 CBF_Constraint_obs3: 0.273018616609797 x1_input_acceleration
   + 0.3535303657716418 x2_input_acceleration
   + 0.091081127244287 x3_input_acceleration >= -0.2097733203478102
 CBF_Constraint_obs1: - 0.426981383390203 x1_input_acceleration
   + 0.3535303657716418 x2_input_acceleration
   + 0.091081127244287 x3_input_acceleration >= -0.3444907412806654
 CBF_Constraint_for_table: x3_input_acceleration >= -0.4664863922150265
 Relaxed_CLF_constraint: 0.00836588324081153 x1_input_acceleration
   + 0.3325084733101678 x2_input_acceleration
   + 0.0462852216154602 x3_input_acceleration + relaxation_CLF
   <= -20.04510967782618
 CBF_Constraint_obs2: - 0.1101373091402539 x1_input_acceleration
   + 0.3501268423266863 x2_input_acceleration
   + 0.0937350626755509 x3_input_acceleration >= -0.129381618196377
 CBF_Constraint_obs3: 0.2698626908597461 x1_input_acceleration
   + 0.3501268423266863 x2_input_acceleration
   + 0.0937350626755509 x3_input_acceleration >= -0.2052511745131358
 CBF_Constraint_obs1: - 0.4301373091402539 x1_input_acceleration
   + 0.3501268423266863 x2_input_acceleration
   + 0.0937350626755509 x3_input_acceleration >= -0.3454914655085801
 CBF_Constraint_for_table: x3_input_acceleration >= -0.4739863922121012
 Relaxed_CLF_constraint: 0.00686588324115225 x1_input_acceleration
   + 0.3310084733102547 x2_input_acceleration
   + 0.0477852216148751 x3_input_acceleration + relaxation_CLF
   <= -20.04475887086426
 CBF_Constraint_obs2: - 0.113137309138812 x1_input_acceleration
   + 0.3471268423272332 x2_input_acceleration
   + 0.0967350626691756 x3_input_acceleration >= -0.1283184596660704
 CBF_Constraint_obs3: 0.266862690861188 x1_input_acceleration
   + 0.3471268423272332 x2_input_acceleration
   + 0.0967350626691756 x3_input_acceleration >= -0.201338015984199
 CBF_Constraint_obs1: - 0.433137309138812 x1_input_acceleration
   + 0.3471268423272332 x2_input_acceleration
   + 0.0967350626691756 x3_input_acceleration >= -0.34682830697712
 CBF_Constraint_for_table: x3_input_acceleration >= -0.4814863921961628
 Relaxed_CLF_constraint: 0.00536588324187323 x1_input_acceleration
   + 0.3295084733105282 x2_input_acceleration
   + 0.0492852216116874 x3_input_acceleration + relaxation_CLF
   <= -20.04441346390228
 CBF_Constraint_obs2: - 0.1161373091387177 x1_input_acceleration
   + 0.3441268423272639 x2_input_acceleration
   + 0.0997350626690337 x3_input_acceleration >= -0.1273228011371396
 CBF_Constraint_obs3: 0.2638626908612823 x1_input_acceleration
   + 0.3441268423272639 x2_input_acceleration
   + 0.0997350626690337 x3_input_acceleration >= -0.1974923574553578
 CBF_Constraint_obs1: - 0.4361373091387177 x1_input_acceleration
   + 0.3441268423272639 x2_input_acceleration
   + 0.0997350626690337 x3_input_acceleration >= -0.3482326484481138
 CBF_Constraint_for_table: x3_input_acceleration >= -0.4889863921958081
 Relaxed_CLF_constraint: 0.00386588324192036 x1_input_acceleration
   + 0.3280084733105435 x2_input_acceleration
   + 0.0507852216116165 x3_input_acceleration + relaxation_CLF
   <= -20.04407345694036
 CBF_Constraint_obs2: - 0.1191373091316668 x1_input_acceleration
   + 0.341126842328065 x2_input_acceleration
   + 0.1027350626633388 x3_input_acceleration >= -0.1263946426053638
 CBF_Constraint_obs3: 0.2608626908683332 x1_input_acceleration
   + 0.341126842328065 x2_input_acceleration
   + 0.1027350626633388 x3_input_acceleration >= -0.1937141989302803
 CBF_Constraint_obs1: - 0.4391373091316668 x1_input_acceleration
   + 0.341126842328065 x2_input_acceleration
   + 0.1027350626633388 x3_input_acceleration >= -0.3497044899106972
 CBF_Constraint_for_table: x3_input_acceleration >= -0.4964324499780492
 Relaxed_CLF_constraint: 0.00236588324544582 x1_input_acceleration
   + 0.3265084733109441 x2_input_acceleration
   + 0.052285221608769 x3_input_acceleration + relaxation_CLF
   <= -20.04373884997842
 CBF_Constraint_obs2: - 0.1219799528690989 x1_input_acceleration
   + 0.3385214250765726 x2_input_acceleration
   + 0.1060503712335177 x3_input_acceleration >= -0.1259031817207351
 CBF_Constraint_obs3: 0.2580200471309012 x1_input_acceleration
   + 0.3385214250765726 x2_input_acceleration
   + 0.1060503712335177 x3_input_acceleration >= -0.1905222264950912
 CBF_Constraint_obs1: - 0.4419799528690989 x1_input_acceleration
   + 0.3385214250765726 x2_input_acceleration
   + 0.1060503712335177 x3_input_acceleration >= -0.3514871440160142
 CBF_Constraint_for_table: x3_input_acceleration >= -0.5039324499763419
 Relaxed_CLF_constraint: 8.6588324562628105e-04 x1_input_acceleration
   + 0.3250084733109913 x2_input_acceleration
   + 0.0537852216084276 x3_input_acceleration + relaxation_CLF
   <= -20.04340964301648
 CBF_Constraint_obs2: - 0.1249799528688451 x1_input_acceleration
   + 0.3355214250766165 x2_input_acceleration
   + 0.1090503712328752 x3_input_acceleration >= -0.1251082484632128
 CBF_Constraint_obs3: 0.2550200471311549 x1_input_acceleration
   + 0.3355214250766165 x2_input_acceleration
   + 0.1090503712328752 x3_input_acceleration >= -0.18687729323781
 CBF_Constraint_obs1: - 0.4449799528688451 x1_input_acceleration
   + 0.3355214250766165 x2_input_acceleration
   + 0.1090503712328752 x3_input_acceleration >= -0.3530922107582889
 CBF_Constraint_for_table: x3_input_acceleration >= -0.5114324499747356
 Relaxed_CLF_constraint: - 6.3411675424682179e-04 x1_input_acceleration
   + 0.3235084733110133 x2_input_acceleration
   + 0.0552852216081063 x3_input_acceleration + relaxation_CLF
   <= -20.04308583605453
 CBF_Constraint_obs2: - 0.1279799528675779 x1_input_acceleration
   + 0.3325214250768982 x2_input_acceleration
   + 0.1120503712290497 x3_input_acceleration >= -0.1243808152046581
 CBF_Constraint_obs3: 0.2520200471324221 x1_input_acceleration
   + 0.3325214250768982 x2_input_acceleration
   + 0.1120503712290497 x3_input_acceleration >= -0.1832998599804591
 CBF_Constraint_obs1: - 0.4479799528675779 x1_input_acceleration
   + 0.3325214250768982 x2_input_acceleration
   + 0.1120503712290497 x3_input_acceleration >= -0.3547647774987205
 CBF_Constraint_for_table: x3_input_acceleration >= -0.5189324499651717
 Relaxed_CLF_constraint: - 0.00213411675361325 x1_input_acceleration
   + 0.3220084733111541 x2_input_acceleration
   + 0.0567852216061936 x3_input_acceleration + relaxation_CLF
   <= -20.04276742909255
 CBF_Constraint_obs2: - 0.1309799528673125 x1_input_acceleration
   + 0.329521425076981 x2_input_acceleration
   + 0.1150503712287326 x3_input_acceleration >= -0.1237208819471962
 CBF_Constraint_obs3: 0.2490200471326875 x1_input_acceleration
   + 0.329521425076981 x2_input_acceleration
   + 0.1150503712287326 x3_input_acceleration >= -0.1797899267232494
 CBF_Constraint_obs1: - 0.4509799528673125 x1_input_acceleration
   + 0.329521425076981 x2_input_acceleration
   + 0.1150503712287326 x3_input_acceleration >= -0.3565048442410462
 CBF_Constraint_for_table: x3_input_acceleration >= -0.526432449964379
 Relaxed_CLF_constraint: - 0.00363411675348052 x1_input_acceleration
   + 0.3205084733111955 x2_input_acceleration
   + 0.058285221606035 x3_input_acceleration + relaxation_CLF
   <= -20.04245442213061
 CBF_Constraint_obs2: - 0.1340414648538761 x1_input_acceleration
   + 0.3264868543844412 x2_input_acceleration
   + 0.1180338011208326 x3_input_acceleration >= -0.1231159482413156
 CBF_Constraint_obs3: 0.2459585351461239 x1_input_acceleration
   + 0.3264868543844412 x2_input_acceleration
   + 0.1180338011208326 x3_input_acceleration >= -0.1762765566301333
 CBF_Constraint_obs1: - 0.4539799528604679 x1_input_acceleration
   + 0.326521425078411 x2_input_acceleration
   + 0.1180503712120802 x3_input_acceleration >= -0.3583124109721737
 CBF_Constraint_for_table: x3_input_acceleration >= -0.5344116712667422
 Relaxed_CLF_constraint: - 0.00513411675005826 x1_input_acceleration
   + 0.3190084733119105 x2_input_acceleration
   + 0.0597852215977088 x3_input_acceleration + relaxation_CLF
   <= -20.04214681516843
 CBF_Constraint_obs2: - 0.1370414648460315 x1_input_acceleration
   + 0.3234868543866402 x2_input_acceleration
   + 0.1210338011153871 x3_input_acceleration >= -0.1225916113256856
 CBF_Constraint_obs3: 0.2429585351539685 x1_input_acceleration
   + 0.3234868543866402 x2_input_acceleration
   + 0.1210338011153871 x3_input_acceleration >= -0.1729022197219557
 CBF_Constraint_obs1: - 0.4570414648460315 x1_input_acceleration
   + 0.3234868543866402 x2_input_acceleration
   + 0.1210338011153871 x3_input_acceleration >= -0.3602247832025108
 CBF_Constraint_for_table: x3_input_acceleration >= -0.5419116712531284
 Relaxed_CLF_constraint: - 0.00663411674613595 x1_input_acceleration
   + 0.31750847331301 x2_input_acceleration
   + 0.0612852215949861 x3_input_acceleration + relaxation_CLF
   <= -20.0418446082066
 CBF_Constraint_obs2: - 0.1400490942284175 x1_input_acceleration
   + 0.3204786289468353 x2_input_acceleration
   + 0.1240357084069748 x3_input_acceleration >= -0.1221314467073295
 CBF_Constraint_obs3: 0.2399585351661138 x1_input_acceleration
   + 0.3204868543878143 x2_input_acceleration
   + 0.1240338010583419 x3_input_acceleration >= -0.1695953827993044
 CBF_Constraint_obs1: - 0.4600414648338862 x1_input_acceleration
   + 0.3204868543878143 x2_input_acceleration
   + 0.1240338010583419 x3_input_acceleration >= -0.3621679462586053
 CBF_Constraint_for_table: x3_input_acceleration >= -0.5494116711105155
 Relaxed_CLF_constraint: - 0.00813411674006331 x1_input_acceleration
   + 0.316008473313597 x2_input_acceleration
   + 0.0627852215664635 x3_input_acceleration + relaxation_CLF
   <= -20.04154780124331
 CBF_Constraint_obs2: - 0.1430490942281414 x1_input_acceleration
   + 0.3174786289469177 x2_input_acceleration
   + 0.1270357084067858 x3_input_acceleration >= -0.1217422430099003
 CBF_Constraint_obs3: 0.2369509057718586 x1_input_acceleration
   + 0.3174786289469177 x2_input_acceleration
   + 0.1270357084067858 x3_input_acceleration >= -0.1663456034931659
 CBF_Constraint_obs1: - 0.4630490942281414 x1_input_acceleration
   + 0.3174786289469177 x2_input_acceleration
   + 0.1270357084067858 x3_input_acceleration >= -0.3641815183924134
 CBF_Constraint_for_table: x3_input_acceleration >= -0.5565153002222868
 Relaxed_CLF_constraint: - 0.00963411673992529 x1_input_acceleration
   + 0.3145084733136382 x2_input_acceleration
   + 0.064285221566369 x3_input_acceleration + relaxation_CLF
   <= -20.04125639428131
 CBF_Constraint_obs2: - 0.1460490941978874 x1_input_acceleration
   + 0.3144786289524657 x2_input_acceleration
   + 0.1300357082846122 x3_input_acceleration >= -0.1214205392661584
 CBF_Constraint_obs3: 0.2339509058021126 x1_input_acceleration
   + 0.3144786289524657 x2_input_acceleration
   + 0.1300357082846122 x3_input_acceleration >= -0.1631738997781654
 CBF_Constraint_obs1: - 0.4660490941978874 x1_input_acceleration
   + 0.3144786289524657 x2_input_acceleration
   + 0.1300357082846122 x3_input_acceleration >= -0.3662598146244683
 CBF_Constraint_for_table: x3_input_acceleration >= -0.5640152999168527
 Relaxed_CLF_constraint: - 0.0111341167247982 x1_input_acceleration
   + 0.3130084733164122 x2_input_acceleration
   + 0.0657852215052822 x3_input_acceleration + relaxation_CLF
   <= -20.04097038731664
 CBF_Constraint_obs2: - 0.1490490941974928 x1_input_acceleration
   + 0.3114786289526181 x2_input_acceleration
   + 0.1330357082837352 x3_input_acceleration >= -0.1211663355673136
 CBF_Constraint_obs3: 0.2309509058025072 x1_input_acceleration
   + 0.3114786289526181 x2_input_acceleration
   + 0.1330357082837352 x3_input_acceleration >= -0.1600696960796955
 CBF_Constraint_obs1: - 0.4690490941974928 x1_input_acceleration
   + 0.3114786289526181 x2_input_acceleration
   + 0.1330357082837352 x3_input_acceleration >= -0.3684056109253078
 CBF_Constraint_for_table: x3_input_acceleration >= -0.5715152999146602
 Relaxed_CLF_constraint: - 0.012634116724601 x1_input_acceleration
   + 0.3115084733164885 x2_input_acceleration
   + 0.0672852215048437 x3_input_acceleration + relaxation_CLF
   <= -20.04068978035453
 CBF_Constraint_obs2: - 0.1520490941926695 x1_input_acceleration
   + 0.3084786289533319 x2_input_acceleration
   + 0.1360357082484237 x3_input_acceleration >= -0.1209796318554862
 CBF_Constraint_obs3: 0.2279509058073305 x1_input_acceleration
   + 0.3084786289533319 x2_input_acceleration
   + 0.1360357082484237 x3_input_acceleration >= -0.1570329923724502
 CBF_Constraint_obs1: - 0.4720490941926695 x1_input_acceleration
   + 0.3084786289533319 x2_input_acceleration
   + 0.1360357082484237 x3_input_acceleration >= -0.3706189072096218
 CBF_Constraint_for_table: x3_input_acceleration >= -0.5788901300723532
 Relaxed_CLF_constraint: - 0.0141341167221893 x1_input_acceleration
   + 0.3100084733168453 x2_input_acceleration
   + 0.068785221487188 x3_input_acceleration + relaxation_CLF
   <= -20.04041457339152
 CBF_Constraint_obs2: - 0.1549842442825303 x1_input_acceleration
   + 0.3055270279398524 x2_input_acceleration
   + 0.1390520398129499 x3_input_acceleration >= -0.1208779381921608
 CBF_Constraint_obs3: 0.2250157557174697 x1_input_acceleration
   + 0.3055270279398524 x2_input_acceleration
   + 0.1390520398129499 x3_input_acceleration >= -0.154142906123757
 CBF_Constraint_obs1: - 0.4749842442825303 x1_input_acceleration
   + 0.3055270279398524 x2_input_acceleration
   + 0.1390520398129499 x3_input_acceleration >= -0.372865333618185
 CBF_Constraint_for_table: x3_input_acceleration >= -0.5863901298019977
 Relaxed_CLF_constraint: - 0.0156341166938775 x1_input_acceleration
   + 0.3085084733243268 x2_input_acceleration
   + 0.0702852214331168 x3_input_acceleration + relaxation_CLF
   <= -20.04014476642784
 CBF_Constraint_obs2: - 0.1579842442824199 x1_input_acceleration
   + 0.3025270279398778 x2_input_acceleration
   + 0.1420520398128219 x3_input_acceleration >= -0.1208255076132581
 CBF_Constraint_obs3: 0.2220157557175801 x1_input_acceleration
   + 0.3025270279398778 x2_input_acceleration
   + 0.1420520398128219 x3_input_acceleration >= -0.1512404755449592
 CBF_Constraint_obs1: - 0.4779842442824199 x1_input_acceleration
   + 0.3025270279398778 x2_input_acceleration
   + 0.1420520398128219 x3_input_acceleration >= -0.375212903039194
 CBF_Constraint_for_table: x3_input_acceleration >= -0.5938901298016775
 Relaxed_CLF_constraint: - 0.0171341166938223 x1_input_acceleration
   + 0.3070084733243395 x2_input_acceleration
   + 0.0717852214330528 x3_input_acceleration + relaxation_CLF
   <= -20.03988035946561
 CBF_Constraint_obs2: - 0.1609842442822952 x1_input_acceleration
   + 0.2995270279399012 x2_input_acceleration
   + 0.1450520398127484 x3_input_acceleration >= -0.120840577034364
 CBF_Constraint_obs3: 0.2190157557177048 x1_input_acceleration
   + 0.2995270279399012 x2_input_acceleration
   + 0.1450520398127484 x3_input_acceleration >= -0.1484055449661835
 CBF_Constraint_obs1: - 0.4809842442822952 x1_input_acceleration
   + 0.2995270279399012 x2_input_acceleration
   + 0.1450520398127484 x3_input_acceleration >= -0.3776279724602002
 CBF_Constraint_for_table: x3_input_acceleration >= -0.6013901298014939
 Relaxed_CLF_constraint: - 0.01863411669376 x1_input_acceleration
   + 0.3055084733243512 x2_input_acceleration
   + 0.0732852214330161 x3_input_acceleration + relaxation_CLF
   <= -20.03962135250337
 CBF_Constraint_obs2: - 0.1639842442722469 x1_input_acceleration
   + 0.2965270279426951 x2_input_acceleration
   + 0.1480520397993066 x3_input_acceleration >= -0.1209231464485041
 CBF_Constraint_obs3: 0.2160157557277531 x1_input_acceleration
   + 0.2965270279426951 x2_input_acceleration
   + 0.1480520397993066 x3_input_acceleration >= -0.1456381143898696
 CBF_Constraint_obs1: - 0.4839842442722469 x1_input_acceleration
   + 0.2965270279426951 x2_input_acceleration
   + 0.1480520397993066 x3_input_acceleration >= -0.3801105418663016
 CBF_Constraint_for_table: x3_input_acceleration >= -0.6088901297678894
 Relaxed_CLF_constraint: - 0.0201341166887358 x1_input_acceleration
   + 0.3040084733257482 x2_input_acceleration
   + 0.0747852214262952 x3_input_acceleration + relaxation_CLF
   <= -20.03936774554099
 CBF_Constraint_obs2: - 0.1671234807222302 x1_input_acceleration
   + 0.2931210011025405 x2_input_acceleration
   + 0.1507378041113936 x3_input_acceleration >= -0.1207150808554925
 CBF_Constraint_obs3: 0.2128765192777698 x1_input_acceleration
   + 0.2931210011025405 x2_input_acceleration
   + 0.1507378041113936 x3_input_acceleration >= -0.1424477741693738
 CBF_Constraint_obs1: - 0.4871234807222302 x1_input_acceleration
   + 0.2931210011025405 x2_input_acceleration
   + 0.1507378041113936 x3_input_acceleration >= -0.3824138654332768
 CBF_Constraint_for_table: x3_input_acceleration >= -0.6164545027826038
 Relaxed_CLF_constraint: - 0.0216341166886298 x1_input_acceleration
   + 0.3025084733257758 x2_input_acceleration
   + 0.0762852214259666 x3_input_acceleration + relaxation_CLF
   <= -20.03911953857872
 CBF_Constraint_obs2: - 0.1699780453889383 x1_input_acceleration
   + 0.2905379951974233 x2_input_acceleration
   + 0.1540854183989914 x3_input_acceleration >= -0.1213089734134677
 CBF_Constraint_obs3: 0.2098765192778097 x1_input_acceleration
   + 0.2901210011025747 x2_input_acceleration
   + 0.1537378041106613 x3_input_acceleration >= -0.1398170762971213
 CBF_Constraint_obs1: - 0.4901234807221903 x1_input_acceleration
   + 0.2901210011025747 x2_input_acceleration
   + 0.1537378041106613 x3_input_acceleration >= -0.3850331675609543
 CBF_Constraint_for_table: x3_input_acceleration >= -0.6239545027807731
 Relaxed_CLF_constraint: - 0.0231341166886099 x1_input_acceleration
   + 0.3010084733257929 x2_input_acceleration
   + 0.0777852214256004 x3_input_acceleration + relaxation_CLF
   <= -20.03887673161645
 CBF_Constraint_obs2: - 0.1729780453887169 x1_input_acceleration
   + 0.2875379951974766 x2_input_acceleration
   + 0.157085418398629 x3_input_acceleration >= -0.1215941644276967
 CBF_Constraint_obs3: 0.2070219546112831 x1_input_acceleration
   + 0.2875379951974766 x2_input_acceleration
   + 0.157085418398629 x3_input_acceleration >= -0.1377650213084157
 CBF_Constraint_obs1: - 0.4929780453887169 x1_input_acceleration
   + 0.2875379951974766 x2_input_acceleration
   + 0.157085418398629 x3_input_acceleration >= -0.3879766007386702
 CBF_Constraint_for_table: x3_input_acceleration >= -0.6314333431309718
 Relaxed_CLF_constraint: - 0.0246341166884991 x1_input_acceleration
   + 0.2995084733258195 x2_input_acceleration
   + 0.0792852214254193 x3_input_acceleration + relaxation_CLF
   <= -20.03863932465418
 CBF_Constraint_obs2: - 0.1759780453828443 x1_input_acceleration
   + 0.2845379951997826 x2_input_acceleration
   + 0.1600854183802307 x3_input_acceleration >= -0.1219468554338143
 CBF_Constraint_obs3: 0.2040219546171557 x1_input_acceleration
   + 0.2845379951997826 x2_input_acceleration
   + 0.1600854183802307 x3_input_acceleration >= -0.1352677123201122
 CBF_Constraint_obs1: - 0.4959780453828443 x1_input_acceleration
   + 0.2845379951997826 x2_input_acceleration
   + 0.1600854183802307 x3_input_acceleration >= -0.3907292917400896
 CBF_Constraint_for_table: x3_input_acceleration >= -0.6389333430849762
 Relaxed_CLF_constraint: - 0.0261341166855628 x1_input_acceleration
   + 0.2980084733269726 x2_input_acceleration
   + 0.0807852214162201 x3_input_acceleration + relaxation_CLF
   <= -20.03840731769155
 CBF_Constraint_obs2: - 0.1789780453827968 x1_input_acceleration
   + 0.2815379951998004 x2_input_acceleration
   + 0.1630854183799523 x3_input_acceleration >= -0.1223670464479167
 CBF_Constraint_obs3: 0.2010219546172032 x1_input_acceleration
   + 0.2815379951998004 x2_input_acceleration
   + 0.1630854183799523 x3_input_acceleration >= -0.1328379033342598
 CBF_Constraint_obs1: - 0.4989780453827968 x1_input_acceleration
   + 0.2815379951998004 x2_input_acceleration
   + 0.1630854183799523 x3_input_acceleration >= -0.3935494827541541
 CBF_Constraint_for_table: x3_input_acceleration >= -0.64643334308428
 Relaxed_CLF_constraint: - 0.0276341166855391 x1_input_acceleration
   + 0.2965084733269814 x2_input_acceleration
   + 0.0822852214160809 x3_input_acceleration + relaxation_CLF
   <= -20.03818071072927
 CBF_Constraint_obs2: - 0.1819780453822475 x1_input_acceleration
   + 0.2785379951999163 x2_input_acceleration
   + 0.1660854183760305 x3_input_acceleration >= -0.1228547374603413
 CBF_Constraint_obs3: 0.1980219546177525 x1_input_acceleration
   + 0.2785379951999163 x2_input_acceleration
   + 0.1660854183760305 x3_input_acceleration >= -0.1304755943472061
 CBF_Constraint_obs1: - 0.5019780453822476 x1_input_acceleration
   + 0.2785379951999163 x2_input_acceleration
   + 0.1660854183760305 x3_input_acceleration >= -0.3964371737661393
 CBF_Constraint_for_table: x3_input_acceleration >= -0.6539694038845647
 Relaxed_CLF_constraint: - 0.0291341166852645 x1_input_acceleration
   + 0.2950084733270394 x2_input_acceleration
   + 0.08378522141412 x3_input_acceleration + relaxation_CLF
   <= -20.03795950376688
 CBF_Constraint_obs2: - 0.1851263417307787 x1_input_acceleration
   + 0.2751187361294829 x2_input_acceleration
   + 0.1687386385278562 x3_input_acceleration >= -0.1230435118804245
 CBF_Constraint_obs3: 0.1948736582692213 x1_input_acceleration
   + 0.2751187361294829 x2_input_acceleration
   + 0.1687386385278562 x3_input_acceleration >= -0.1276734872361847
 CBF_Constraint_obs1: - 0.5051263417307787 x1_input_acceleration
   + 0.2751187361294829 x2_input_acceleration
   + 0.1687386385278562 x3_input_acceleration >= -0.3991445852650476
 CBF_Constraint_for_table: x3_input_acceleration >= -0.6614694038223869
 Relaxed_CLF_constraint: - 0.0306341166814295 x1_input_acceleration
   + 0.2935084733274977 x2_input_acceleration
   + 0.0852852214016845 x3_input_acceleration + relaxation_CLF
   <= -20.03774369680377
 CBF_Constraint_obs2: - 0.1881263417232248 x1_input_acceleration
   + 0.2721187361335956 x2_input_acceleration
   + 0.1717386383270618 x3_input_acceleration >= -0.1236678586244279
 CBF_Constraint_obs3: 0.1918736582767752 x1_input_acceleration
   + 0.2721187361335956 x2_input_acceleration
   + 0.1717386383270618 x3_input_acceleration >= -0.1254478339873643
 CBF_Constraint_obs1: - 0.5081263417232249 x1_input_acceleration
   + 0.2721187361335956 x2_input_acceleration
   + 0.1717386383270618 x3_input_acceleration >= -0.4021689320030078
 CBF_Constraint_for_table: x3_input_acceleration >= -0.668969403320401
 Relaxed_CLF_constraint: - 0.0321341166776525 x1_input_acceleration
   + 0.2920084733295541 x2_input_acceleration
   + 0.0867852213012873 x3_input_acceleration + relaxation_CLF
   <= -20.03753328983489
 CBF_Constraint_obs2: - 0.1911263417038565 x1_input_acceleration
   + 0.2691187361348313 x2_input_acceleration
   + 0.1747386382402384 x3_input_acceleration >= -0.1243597054074514
 CBF_Constraint_obs3: 0.1888736582961435 x1_input_acceleration
   + 0.2691187361348313 x2_input_acceleration
   + 0.1747386382402384 x3_input_acceleration >= -0.1232896807887878
 CBF_Constraint_obs1: - 0.5111263417038565 x1_input_acceleration
   + 0.2691187361348313 x2_input_acceleration
   + 0.1747386382402384 x3_input_acceleration >= -0.4052607787705366
 CBF_Constraint_for_table: x3_input_acceleration >= -0.6764694031033425
 Relaxed_CLF_constraint: - 0.0336341166679684 x1_input_acceleration
   + 0.2905084733301719 x2_input_acceleration
   + 0.0882852212578756 x3_input_acceleration + relaxation_CLF
   <= -20.03732828286929
 CBF_Constraint_obs2: - 0.1941263416827145 x1_input_acceleration
   + 0.2661187361361372 x2_input_acceleration
   + 0.1777386381356399 x3_input_acceleration >= -0.1251190521801512
 CBF_Constraint_obs3: 0.1858736583172855 x1_input_acceleration
   + 0.2661187361361372 x2_input_acceleration
   + 0.1777386381356399 x3_input_acceleration >= -0.1211990275815725
 CBF_Constraint_obs1: - 0.5141263416827144 x1_input_acceleration
   + 0.2661187361361372 x2_input_acceleration
   + 0.1777386381356399 x3_input_acceleration >= -0.4084201255263227
 CBF_Constraint_for_table: x3_input_acceleration >= -0.6839694028418464
 Relaxed_CLF_constraint: - 0.0351341166573974 x1_input_acceleration
   + 0.2890084733308249 x2_input_acceleration
   + 0.0897852212055763 x3_input_acceleration + relaxation_CLF
   <= -20.0371286759029
 CBF_Constraint_obs2: - 0.1970524319229897 x1_input_acceleration
   + 0.2631869238498096 x2_input_acceleration
   + 0.1807928783616966 x3_input_acceleration >= -0.1259788533482465
 CBF_Constraint_obs3: 0.1828736583174888 x1_input_acceleration
   + 0.2631187361361865 x2_input_acceleration
   + 0.180738638134951 x3_input_acceleration >= -0.1191758744090032
 CBF_Constraint_obs1: - 0.5171263416825111 x1_input_acceleration
   + 0.2631187361361865 x2_input_acceleration
   + 0.180738638134951 x3_input_acceleration >= -0.4116469723533978
 CBF_Constraint_for_table: x3_input_acceleration >= -0.6914694028401241
 Relaxed_CLF_constraint: - 0.0366341166572957 x1_input_acceleration
   + 0.2875084733308496 x2_input_acceleration
   + 0.0912852212052319 x3_input_acceleration + relaxation_CLF
   <= -20.03693446894031
 CBF_Constraint_obs2: - 0.2000524319229186 x1_input_acceleration
   + 0.2601869238498269 x2_input_acceleration
   + 0.1837928783614756 x3_input_acceleration >= -0.1268725412463822
 CBF_Constraint_obs3: 0.1799475680770814 x1_input_acceleration
   + 0.2601869238498269 x2_input_acceleration
   + 0.1837928783614756 x3_input_acceleration >= -0.1173227309196095
 CBF_Constraint_obs1: - 0.5201263416824401 x1_input_acceleration
   + 0.2601187361362038 x2_input_acceleration
   + 0.18373863813473 x3_input_acceleration >= -0.4149413191808246
 CBF_Constraint_for_table: x3_input_acceleration >= -0.6989419847029749
 Relaxed_CLF_constraint: - 0.0381341166572602 x1_input_acceleration
   + 0.2860084733308582 x2_input_acceleration
   + 0.0927852212051214 x3_input_acceleration + relaxation_CLF
   <= -20.03674566197774
 CBF_Constraint_obs2: - 0.2030524319216171 x1_input_acceleration
   + 0.2571869238500125 x2_input_acceleration
   + 0.1867928783603767 x3_input_acceleration >= -0.127833729143587
 CBF_Constraint_obs3: 0.1769475680783829 x1_input_acceleration
   + 0.2571869238500125 x2_input_acceleration
   + 0.1867928783603767 x3_input_acceleration >= -0.1154339188180507
 CBF_Constraint_obs1: - 0.5231263416811387 x1_input_acceleration
   + 0.2571187361363895 x2_input_acceleration
   + 0.1867386381336311 x3_input_acceleration >= -0.4183031660063362
 CBF_Constraint_for_table: x3_input_acceleration >= -0.70692567639258
 Relaxed_CLF_constraint: - 0.0396341166566094 x1_input_acceleration
   + 0.284508473330951 x2_input_acceleration
   + 0.0942852212045719 x3_input_acceleration + relaxation_CLF
   <= -20.03656225501514
 CBF_Constraint_obs2: - 0.206022391157378 x1_input_acceleration
   + 0.2544348791735209 x2_input_acceleration
   + 0.1900621719901641 x3_input_acceleration >= -0.1291324532748332
 CBF_Constraint_obs3: 0.1739475681016552 x1_input_acceleration
   + 0.2541869238512553 x2_input_acceleration
   + 0.1897928782050689 x3_input_acceleration >= -0.1136126066535099
 CBF_Constraint_obs1: - 0.5260524318983448 x1_input_acceleration
   + 0.2541869238512553 x2_input_acceleration
   + 0.1897928782050689 x3_input_acceleration >= -0.4217043624756134
 CBF_Constraint_for_table: x3_input_acceleration >= -0.7144256760043105
 Relaxed_CLF_constraint: - 0.0411341166449732 x1_input_acceleration
   + 0.2830084733315724 x2_input_acceleration
   + 0.095785221126918 x3_input_acceleration + relaxation_CLF
   <= -20.03638424804639
 CBF_Constraint_obs2: - 0.2090223911531084 x1_input_acceleration
   + 0.2514348791740852 x2_input_acceleration
   + 0.1930621719726071 x3_input_acceleration >= -0.130228575894288
 CBF_Constraint_obs3: 0.1709776088468916 x1_input_acceleration
   + 0.2514348791740852 x2_input_acceleration
   + 0.1930621719726071 x3_input_acceleration >= -0.112157304298835
 CBF_Constraint_obs1: - 0.5290223911531085 x1_input_acceleration
   + 0.2514348791740852 x2_input_acceleration
   + 0.1930621719726071 x3_input_acceleration >= -0.4254464888167748
 CBF_Constraint_for_table: x3_input_acceleration >= -0.7214667201956473
 Relaxed_CLF_constraint: - 0.0426341166428385 x1_input_acceleration
   + 0.2815084733318546 x2_input_acceleration
   + 0.0972852211181395 x3_input_acceleration + relaxation_CLF
   <= -20.03621164108302
 CBF_Constraint_obs2: - 0.212041464639165 x1_input_acceleration
   + 0.2484876888894094 x2_input_acceleration
   + 0.1960399990385328 x3_input_acceleration >= -0.1314242443486876
 CBF_Constraint_obs3: 0.1679776088471631 x1_input_acceleration
   + 0.2484348791741384 x2_input_acceleration
   + 0.1960621719663892 x3_input_acceleration >= -0.1104709269255716
 CBF_Constraint_obs1: - 0.5320223911528369 x1_input_acceleration
   + 0.2484348791741384 x2_input_acceleration
   + 0.1960621719663892 x3_input_acceleration >= -0.4290101114430361
 CBF_Constraint_for_table: x3_input_acceleration >= -0.7289667201801026
 Relaxed_CLF_constraint: - 0.0441341166427027 x1_input_acceleration
   + 0.2800084733318812 x2_input_acceleration
   + 0.0987852211150306 x3_input_acceleration + relaxation_CLF
   <= -20.03604443412009
 CBF_Constraint_obs2: - 0.2150414646372611 x1_input_acceleration
   + 0.2454876888897743 x2_input_acceleration
   + 0.1990399990315684 x3_input_acceleration >= -0.1326549476553347
 CBF_Constraint_obs3: 0.1649585353627389 x1_input_acceleration
   + 0.2454876888897743 x2_input_acceleration
   + 0.1990399990315684 x3_input_acceleration >= -0.1088655562499367
 CBF_Constraint_obs1: - 0.5350414646372611 x1_input_acceleration
   + 0.2454876888897743 x2_input_acceleration
   + 0.1990399990315684 x3_input_acceleration >= -0.4326881193651436
 CBF_Constraint_for_table: x3_input_acceleration >= -0.7364667201626918
 Relaxed_CLF_constraint: - 0.0456341166417507 x1_input_acceleration
   + 0.2785084733320636 x2_input_acceleration
   + 0.1002852211115484 x3_input_acceleration + relaxation_CLF
   <= -20.03588262715713
 CBF_Constraint_obs2: - 0.2180414646362931 x1_input_acceleration
   + 0.2424876888899942 x2_input_acceleration
   + 0.2020399990233651 x3_input_acceleration >= -0.1339531509616398
 CBF_Constraint_obs3: 0.1619585353637069 x1_input_acceleration
   + 0.2424876888899942 x2_input_acceleration
   + 0.2020399990233651 x3_input_acceleration >= -0.1073137595571613
 CBF_Constraint_obs1: - 0.5380414646362931 x1_input_acceleration
   + 0.2424876888899942 x2_input_acceleration
   + 0.2020399990233651 x3_input_acceleration >= -0.4363863226706742
 CBF_Constraint_for_table: x3_input_acceleration >= -0.7439667201421835
 Relaxed_CLF_constraint: - 0.0471341166412668 x1_input_acceleration
   + 0.2770084733321735 x2_input_acceleration
   + 0.1017852211074467 x3_input_acceleration + relaxation_CLF
   <= -20.03572622019411
 CBF_Constraint_obs2: - 0.221126818487124 x1_input_acceleration
   + 0.2391180208831922 x2_input_acceleration
   + 0.2047394723990461 x3_input_acceleration >= -0.134990936654465
 CBF_Constraint_obs3: 0.158873181512876 x1_input_acceleration
   + 0.2391180208831922 x2_input_acceleration
   + 0.2047394723990461 x3_input_acceleration >= -0.1054204590916972
 CBF_Constraint_obs1: - 0.541126818487124 x1_input_acceleration
   + 0.2391180208831922 x2_input_acceleration
   + 0.2047394723990461 x3_input_acceleration >= -0.4398923914441641
 CBF_Constraint_for_table: x3_input_acceleration >= -0.7514020490891984
 Relaxed_CLF_constraint: - 0.048634116641023 x1_input_acceleration
   + 0.2755084733322211 x2_input_acceleration
   + 0.103285221104766 x3_input_acceleration + relaxation_CLF
   <= -20.03557521323119
 CBF_Constraint_obs2: - 0.2240457561698261 x1_input_acceleration
   + 0.2361930035263966 x2_input_acceleration
   + 0.2078212499610904 x3_input_acceleration >= -0.1364666346348995
 CBF_Constraint_obs3: 0.1559542438301739 x1_input_acceleration
   + 0.2361930035263966 x2_input_acceleration
   + 0.2078212499610904 x3_input_acceleration >= -0.1041231662735647
 CBF_Constraint_obs1: - 0.5440457561698261 x1_input_acceleration
   + 0.2361930035263966 x2_input_acceleration
   + 0.2078212499610904 x3_input_acceleration >= -0.4437032395707603
 CBF_Constraint_for_table: x3_input_acceleration >= -0.7594003438930519
 Relaxed_CLF_constraint: - 0.0501341166408213 x1_input_acceleration
   + 0.2740084733322595 x2_input_acceleration
   + 0.1047852210994722 x3_input_acceleration + relaxation_CLF
   <= -20.03542960626804
 CBF_Constraint_obs2: - 0.2270457560602356 x1_input_acceleration
   + 0.2331930035330991 x2_input_acceleration
   + 0.210781389248958 x3_input_acceleration >= -0.1379469328693409
 CBF_Constraint_obs3: 0.1529542439397645 x1_input_acceleration
   + 0.2331930035330991 x2_input_acceleration
   + 0.210781389248958 x3_input_acceleration >= -0.1027534646121171
 CBF_Constraint_obs1: - 0.5470457560602355 x1_input_acceleration
   + 0.2331930035330991 x2_input_acceleration
   + 0.210781389248958 x3_input_acceleration >= -0.4475835377175292
 CBF_Constraint_for_table: x3_input_acceleration >= -0.7668006921127211
 Relaxed_CLF_constraint: - 0.0516341165860261 x1_input_acceleration
   + 0.2725084733356108 x2_input_acceleration
   + 0.106265290743406 x3_input_acceleration + relaxation_CLF
   <= -20.03528770482084
 CBF_Constraint_obs2: - 0.2300457560564169 x1_input_acceleration
   + 0.2301930035333537 x2_input_acceleration
   + 0.2136881645712853 x3_input_acceleration >= -0.1394656255413997
 CBF_Constraint_obs3: 0.1499542439435831 x1_input_acceleration
   + 0.2301930035333537 x2_input_acceleration
   + 0.2136881645712853 x3_input_acceleration >= -0.1014221572878036
 CBF_Constraint_obs1: - 0.5500457560564169 x1_input_acceleration
   + 0.2301930035333537 x2_input_acceleration
   + 0.2136881645712853 x3_input_acceleration >= -0.4515022303865332
 CBF_Constraint_for_table: x3_input_acceleration >= -0.774067630418539
 Relaxed_CLF_constraint: - 0.0531341165841167 x1_input_acceleration
   + 0.2710084733357381 x2_input_acceleration
   + 0.1077186784045697 x3_input_acceleration + relaxation_CLF
   <= -20.03514885625687
 CBF_Constraint_obs2: - 0.2330457560027028 x1_input_acceleration
   + 0.2271930035398428 x2_input_acceleration
   + 0.2165424023773825 x3_input_acceleration >= -0.1410219965946182
 CBF_Constraint_obs3: 0.1469542439972972 x1_input_acceleration
   + 0.2271930035398428 x2_input_acceleration
   + 0.2165424023773825 x3_input_acceleration >= -0.1001285283920506
 CBF_Constraint_obs1: - 0.5530457560027028 x1_input_acceleration
   + 0.2271930035398428 x2_input_acceleration
   + 0.2165424023773825 x3_input_acceleration >= -0.4554586013967804
 CBF_Constraint_for_table: x3_input_acceleration >= -0.7812032249337821
 Relaxed_CLF_constraint: - 0.0546341165572597 x1_input_acceleration
   + 0.2695084733389826 x2_input_acceleration
   + 0.1091457973076183 x3_input_acceleration + relaxation_CLF
   <= -20.03501300358537
 CBF_Constraint_obs2: - 0.2357734819502803 x1_input_acceleration
   + 0.2242243555871104 x2_input_acceleration
   + 0.2194047304148538 x3_input_acceleration >= -0.142505165197286
 CBF_Constraint_obs3: 0.1442265180497197 x1_input_acceleration
   + 0.2242243555871104 x2_input_acceleration
   + 0.2194047304148538 x3_input_acceleration >= -0.0990203573445196
 CBF_Constraint_obs1: - 0.5557734819502803 x1_input_acceleration
   + 0.2242243555871104 x2_input_acceleration
   + 0.2194047304148538 x3_input_acceleration >= -0.4591239507575101
 CBF_Constraint_for_table: x3_input_acceleration >= -0.7876938571917669
 Relaxed_CLF_constraint: - 0.0561341165397154 x1_input_acceleration
   + 0.2680084733410405 x2_input_acceleration
   + 0.1105470397946767 x3_input_acceleration + relaxation_CLF
   <= -20.03488009153186
 CBF_Constraint_obs2: - 0.2387734819438434 x1_input_acceleration
   + 0.2212243555890111 x2_input_acceleration
   + 0.2221562083881001 x3_input_acceleration >= -0.1441329651384213
 CBF_Constraint_obs3: 0.1412265180561566 x1_input_acceleration
   + 0.2212243555890111 x2_input_acceleration
   + 0.2221562083881001 x3_input_acceleration >= -0.0977981572917701
 CBF_Constraint_obs1: - 0.5587734819438435 x1_input_acceleration
   + 0.2212243555890111 x2_input_acceleration
   + 0.2221562083881001 x3_input_acceleration >= -0.4631517506934961
 CBF_Constraint_for_table: x3_input_acceleration >= -0.7945725521248828
 Relaxed_CLF_constraint: - 0.057634116536497 x1_input_acceleration
   + 0.2665084733419908 x2_input_acceleration
   + 0.1119227787812998 x3_input_acceleration + relaxation_CLF
   <= -20.03475006646486
 CBF_Constraint_obs2: - 0.2417734819431409 x1_input_acceleration
   + 0.2182243555890657 x2_input_acceleration
   + 0.2248573881639915 x3_input_acceleration >= -0.1457964136938815
 CBF_Constraint_obs3: 0.1382265180568591 x1_input_acceleration
   + 0.2182243555890657 x2_input_acceleration
   + 0.2248573881639915 x3_input_acceleration >= -0.0966116058478977
 CBF_Constraint_obs1: - 0.5617734819431408 x1_input_acceleration
   + 0.2182243555890657 x2_input_acceleration
   + 0.2248573881639915 x3_input_acceleration >= -0.4672151992483942
 CBF_Constraint_for_table: x3_input_acceleration >= -0.8013255015646115
 Relaxed_CLF_constraint: - 0.0591341165361457 x1_input_acceleration
   + 0.2650084733420182 x2_input_acceleration
   + 0.1132733686692455 x3_input_acceleration + relaxation_CLF
   <= -20.0346228762925
 CBF_Constraint_obs2: - 0.2447734819404331 x1_input_acceleration
   + 0.2152243555892019 x2_input_acceleration
   + 0.2275089439588711 x3_input_acceleration >= -0.1474948753516394
 CBF_Constraint_obs3: 0.1352265180595669 x1_input_acceleration
   + 0.2152243555892019 x2_input_acceleration
   + 0.2275089439588711 x3_input_acceleration >= -0.0954600675082279
 CBF_Constraint_obs1: - 0.5647734819404331 x1_input_acceleration
   + 0.2152243555892019 x2_input_acceleration
   + 0.2275089439588711 x3_input_acceleration >= -0.4713136609039859
 CBF_Constraint_for_table: x3_input_acceleration >= -0.8080479703441077
 Relaxed_CLF_constraint: - 0.0606341165347918 x1_input_acceleration
   + 0.2635084733420863 x2_input_acceleration
   + 0.1145991465666853 x3_input_acceleration + relaxation_CLF
   <= -20.03449847040194
 CBF_Constraint_obs2: - 0.2480162440532585 x1_input_acceleration
   + 0.2121639164794865 x2_input_acceleration
   + 0.2300461799677474 x3_input_acceleration >= -0.1493085371099415
 CBF_Constraint_obs3: 0.1319837559467415 x1_input_acceleration
   + 0.2121639164794865 x2_input_acceleration
   + 0.2300461799677474 x3_input_acceleration >= -0.094193105259346
 CBF_Constraint_obs1: - 0.5680162440532585 x1_input_acceleration
   + 0.2121639164794865 x2_input_acceleration
   + 0.2300461799677474 x3_input_acceleration >= -0.4757215323525483
 CBF_Constraint_for_table: x3_input_acceleration >= -0.8150356845995443
 Relaxed_CLF_constraint: - 0.0621131616994809 x1_input_acceleration
   + 0.2620084733421297 x2_input_acceleration
   + 0.1159214683560661 x3_input_acceleration + relaxation_CLF
   <= -20.03437770871408
 CBF_Constraint_obs2: - 0.2509199240452299 x1_input_acceleration
   + 0.2094949606766094 x2_input_acceleration
   + 0.2329719190507587 x3_input_acceleration >= -0.1514060773724392
 CBF_Constraint_obs3: 0.1290800759547701 x1_input_acceleration
   + 0.2094949606766094 x2_input_acceleration
   + 0.2329719190507587 x3_input_acceleration >= -0.0935321495294708
 CBF_Constraint_obs1: - 0.5709199240452298 x1_input_acceleration
   + 0.2094949606766094 x2_input_acceleration
   + 0.2329719190507587 x3_input_acceleration >= -0.480142016608623
 CBF_Constraint_for_table: x3_input_acceleration >= -0.8216472935451707
 Relaxed_CLF_constraint: - 0.0635681011369949 x1_input_acceleration
   + 0.2605084733421499 x2_input_acceleration
   + 0.1172437901451914 x3_input_acceleration + relaxation_CLF
   <= -20.03426066979713
 CBF_Constraint_obs2: - 0.2537821376012573 x1_input_acceleration
   + 0.2064949606774886 x2_input_acceleration
   + 0.2356165626363975 x3_input_acceleration >= -0.1532008834240656
 CBF_Constraint_obs3: 0.1262178623987427 x1_input_acceleration
   + 0.2064949606774886 x2_input_acceleration
   + 0.2356165626363975 x3_input_acceleration >= -0.0926078527028711
 CBF_Constraint_obs1: - 0.5737821376012573 x1_input_acceleration
   + 0.2064949606774886 x2_input_acceleration
   + 0.2356165626363975 x3_input_acceleration >= -0.4842265935050715
 CBF_Constraint_for_table: x3_input_acceleration >= -0.8282589025092677
 Relaxed_CLF_constraint: - 0.0649992079150086 x1_input_acceleration
   + 0.2590084733425895 x2_input_acceleration
   + 0.1185661119380108 x3_input_acceleration + relaxation_CLF
   <= -20.03414728367718
 CBF_Constraint_obs2: - 0.256597207642673 x1_input_acceleration
   + 0.2034949606787921 x2_input_acceleration
   + 0.2382612062163723 x3_input_acceleration >= -0.1550259104742011
 CBF_Constraint_obs3: 0.123402792357327 x1_input_acceleration
   + 0.2034949606787921 x2_input_acceleration
   + 0.2382612062163723 x3_input_acceleration >= -0.0917585632136618
 CBF_Constraint_obs1: - 0.576597207642673 x1_input_acceleration
   + 0.2034949606787921 x2_input_acceleration
   + 0.2382612062163723 x3_input_acceleration >= -0.4883036765883395
 CBF_Constraint_for_table: x3_input_acceleration >= -0.8348705114592048
 Relaxed_CLF_constraint: - 0.0664067429357164 x1_input_acceleration
   + 0.2575084733432412 x2_input_acceleration
   + 0.1198884337279982 x3_input_acceleration + relaxation_CLF
   <= -20.03403748235706
 CBF_Constraint_obs2: - 0.2593451288005588 x1_input_acceleration
   + 0.2004788674269462 x2_input_acceleration
   + 0.2408884452880833 x3_input_acceleration >= -0.1568486439883493
 CBF_Constraint_obs3: 0.1206343672016385 x1_input_acceleration
   + 0.2004949606810356 x2_input_acceleration
   + 0.2409058498443578 x3_input_acceleration >= -0.0909831353723275
 CBF_Constraint_obs1: - 0.5793656327983615 x1_input_acceleration
   + 0.2004949606810356 x2_input_acceleration
   + 0.2409058498443578 x3_input_acceleration >= -0.4923729927694601
 CBF_Constraint_for_table: x3_input_acceleration >= -0.8414821205291684
 Relaxed_CLF_constraint: - 0.0677909555135607 x1_input_acceleration
   + 0.2560084733443629 x2_input_acceleration
   + 0.1212107555419909 x3_input_acceleration + relaxation_CLF
   <= -20.03393119973305
 CBF_Constraint_obs2: - 0.2620673858778927 x1_input_acceleration
   + 0.1974788674282921 x2_input_acceleration
   + 0.2435330888733057 x3_input_acceleration >= -0.1587319789972583
 CBF_Constraint_obs3: 0.1179121101243046 x1_input_acceleration
   + 0.1974949606823815 x2_input_acceleration
   + 0.2435504934295801 x3_input_acceleration >= -0.0902804600733668
 CBF_Constraint_obs1: - 0.5820878898756955 x1_input_acceleration
   + 0.1974949606823815 x2_input_acceleration
   + 0.2435504934295801 x3_input_acceleration >= -0.4964342673558339
 CBF_Constraint_for_table: x3_input_acceleration >= -0.8476759009323487
 Relaxed_CLF_constraint: - 0.0691520840522277 x1_input_acceleration
   + 0.2545084733450359 x2_input_acceleration
   + 0.1225330773346021 x3_input_acceleration + relaxation_CLF
   <= -20.03382837150971
 CBF_Constraint_obs2: - 0.2647439310386965 x1_input_acceleration
   + 0.1944788674418363 x2_input_acceleration
   + 0.2461777324691372 x3_input_acceleration >= -0.1606435685836594
 CBF_Constraint_obs3: 0.1152560689613034 x1_input_acceleration
   + 0.1944788674418363 x2_input_acceleration
   + 0.2461777324691372 x3_input_acceleration >= -0.0896368340968977
 CBF_Constraint_obs1: - 0.5847439310386966 x1_input_acceleration
   + 0.1944788674418363 x2_input_acceleration
   + 0.2461777324691372 x3_input_acceleration >= -0.5004387134146167
 CBF_Constraint_for_table: x3_input_acceleration >= -0.8542875099219274
 Relaxed_CLF_constraint: - 0.0704903566326296 x1_input_acceleration
   + 0.253008473351808 x2_input_acceleration
   + 0.1238553991325178 x3_input_acceleration + relaxation_CLF
   <= -20.03372893514412
 CBF_Constraint_obs2: - 0.2674085785827662 x1_input_acceleration
   + 0.1911850165432206 x2_input_acceleration
   + 0.2485849111433186 x3_input_acceleration >= -0.1623168956230233
 CBF_Constraint_obs3: 0.112624800018308 x1_input_acceleration
   + 0.1914788674419633 x2_input_acceleration
   + 0.2488223760481038 x3_input_acceleration >= -0.089076346347806
 CBF_Constraint_obs1: - 0.5873751999816921 x1_input_acceleration
   + 0.1914788674419633 x2_input_acceleration
   + 0.2488223760481038 x3_input_acceleration >= -0.5044829463157672
 CBF_Constraint_for_table: x3_input_acceleration >= -0.8608991188693438
 Relaxed_CLF_constraint: - 0.0718059911041273 x1_input_acceleration
   + 0.2515084733518715 x2_input_acceleration
   + 0.1251777209220011 x3_input_acceleration + relaxation_CLF
   <= -20.03363282973659
 CBF_Constraint_obs2: - 0.2699949889748428 x1_input_acceleration
   + 0.1881850165436338 x2_input_acceleration
   + 0.2512295547240015 x3_input_acceleration >= -0.1642839796123416
 CBF_Constraint_obs3: 0.1100050110251573 x1_input_acceleration
   + 0.1881850165436338 x2_input_acceleration
   + 0.2512295547240015 x3_input_acceleration >= -0.088288740086241
 CBF_Constraint_obs1: - 0.5899949889748427 x1_input_acceleration
   + 0.1881850165436338 x2_input_acceleration
   + 0.2512295547240015 x3_input_acceleration >= -0.5082799707922158
 CBF_Constraint_for_table: x3_input_acceleration >= -0.8675664581639162
 Relaxed_CLF_constraint: - 0.0730991963001656 x1_input_acceleration
   + 0.2500084733520781 x2_input_acceleration
   + 0.1265000427123426 x3_input_acceleration + relaxation_CLF
   <= -20.03353999602152
 CBF_Constraint_obs2: - 0.2725336024238284 x1_input_acceleration
   + 0.1851847781251117 x2_input_acceleration
   + 0.2538782514184085 x3_input_acceleration >= -0.166277666303281
 CBF_Constraint_obs3: 0.1074663975761716 x1_input_acceleration
   + 0.1851847781251117 x2_input_acceleration
   + 0.2538782514184085 x3_input_acceleration >= -0.087870744000644
 CBF_Constraint_obs1: - 0.5925336024238285 x1_input_acceleration
   + 0.1851847781251117 x2_input_acceleration
   + 0.2538782514184085 x3_input_acceleration >= -0.5123045482423438
 CBF_Constraint_for_table: x3_input_acceleration >= -0.8745607289297797
 Relaxed_CLF_constraint: - 0.0743701719547121 x1_input_acceleration
   + 0.2485084733521066 x2_input_acceleration
   + 0.1278223645016237 x3_input_acceleration + relaxation_CLF
   <= -20.03345037626846
 CBF_Constraint_obs2: - 0.2750314770866243 x1_input_acceleration
   + 0.1821847781251542 x2_input_acceleration
   + 0.2565228949968947 x3_input_acceleration >= -0.1682970030331873
 CBF_Constraint_obs3: 0.1049685229133757 x1_input_acceleration
   + 0.1821847781251542 x2_input_acceleration
   + 0.2565228949968947 x3_input_acceleration >= -0.0875170998008942
 CBF_Constraint_obs1: - 0.5950314770866243 x1_input_acceleration
   + 0.1821847781251542 x2_input_acceleration
   + 0.2565228949968947 x3_input_acceleration >= -0.5163221847024868
 CBF_Constraint_for_table: x3_input_acceleration >= -0.8811723378759952
 Relaxed_CLF_constraint: - 0.0756191092861101 x1_input_acceleration
   + 0.2470084733521278 x2_input_acceleration
   + 0.1291446862908668 x3_input_acceleration + relaxation_CLF
   <= -20.03336391423766
 CBF_Constraint_obs2: - 0.2774856412192284 x1_input_acceleration
   + 0.1791847781269338 x2_input_acceleration
   + 0.2591675385855621 x3_input_acceleration >= -0.1703415985646796
 CBF_Constraint_obs3: 0.1025143587807716 x1_input_acceleration
   + 0.1791847781269338 x2_input_acceleration
   + 0.2591675385855621 x3_input_acceleration >= -0.0872302394064126
 CBF_Constraint_obs1: - 0.5974856412192284 x1_input_acceleration
   + 0.1791847781269338 x2_input_acceleration
   + 0.2591675385855621 x3_input_acceleration >= -0.5203301115400623
 CBF_Constraint_for_table: x3_input_acceleration >= -0.8877839468476639
 Relaxed_CLF_constraint: - 0.0768461913524121 x1_input_acceleration
   + 0.2455084733530176 x2_input_acceleration
   + 0.1304670080852005 x3_input_acceleration + relaxation_CLF
   <= -20.03328055512488
 CBF_Constraint_obs2: - 0.279967970879741 x1_input_acceleration
   + 0.1761405514813811 x2_input_acceleration
   + 0.2617555577743241 x3_input_acceleration >= -0.1724044132755401
 CBF_Constraint_obs3: 0.100032029120259 x1_input_acceleration
   + 0.1761405514813811 x2_input_acceleration
   + 0.2617555577743241 x3_input_acceleration >= -0.0869348409397861
 CBF_Constraint_obs1: - 0.5998964453060106 x1_input_acceleration
   + 0.1761847781278044 x2_input_acceleration
   + 0.2618121821868607 x3_input_acceleration >= -0.5243280498452213
 CBF_Constraint_for_table: x3_input_acceleration >= -0.8943535345763437
 Relaxed_CLF_constraint: - 0.0780515933958032 x1_input_acceleration
   + 0.2440084733534529 x2_input_acceleration
   + 0.1317893298858498 x3_input_acceleration + relaxation_CLF
   <= -20.03320024550867
 CBF_Constraint_obs2: - 0.282335750654143 x1_input_acceleration
   + 0.1731405514831138 x2_input_acceleration
   + 0.2644002013628903 x3_input_acceleration >= -0.1744982414325652
 CBF_Constraint_obs3: 0.097664249345857 x1_input_acceleration
   + 0.1731405514831138 x2_input_acceleration
   + 0.2644002013628903 x3_input_acceleration >= -0.0867792783111293
 CBF_Constraint_obs1: - 0.602335750654143 x1_input_acceleration
   + 0.1731405514831138 x2_input_acceleration
   + 0.2644002013628903 x3_input_acceleration >= -0.5283668419558796
 CBF_Constraint_for_table: x3_input_acceleration >= -0.9009651435477593
 Relaxed_CLF_constraint: - 0.0792354832830042 x1_input_acceleration
   + 0.2425084733543193 x2_input_acceleration
   + 0.1331116516801329 x3_input_acceleration + relaxation_CLF
   <= -20.0331229333091
 CBF_Constraint_obs2: - 0.2846608272045589 x1_input_acceleration
   + 0.1701405514953995 x2_input_acceleration
   + 0.2670448450141145 x3_input_acceleration >= -0.1766156788206935
 CBF_Constraint_obs3: 0.0953391727954412 x1_input_acceleration
   + 0.1701405514953995 x2_input_acceleration
   + 0.2670448450141145 x3_input_acceleration >= -0.0866878929763626
 CBF_Constraint_obs1: - 0.6046608272045588 x1_input_acceleration
   + 0.1701405514953995 x2_input_acceleration
   + 0.2670448450141145 x3_input_acceleration >= -0.5323443405843407
 CBF_Constraint_for_table: x3_input_acceleration >= -0.9075767526758196
 Relaxed_CLF_constraint: - 0.0803980215582121 x1_input_acceleration
   + 0.2410084733604622 x2_input_acceleration
   + 0.134433973505745 x3_input_acceleration + relaxation_CLF
   <= -20.03304856773382
 CBF_Constraint_obs2: - 0.2869435087810714 x1_input_acceleration
   + 0.1671405515136195 x2_input_acceleration
   + 0.2696894887014491 x3_input_acceleration >= -0.1787562018848984
 CBF_Constraint_obs3: 0.0930564912189285 x1_input_acceleration
   + 0.1671405515136195 x2_input_acceleration
   + 0.2696894887014491 x3_input_acceleration >= -0.0866598685428806
 CBF_Constraint_obs1: - 0.6069435087810715 x1_input_acceleration
   + 0.1671405515136195 x2_input_acceleration
   + 0.2696894887014491 x3_input_acceleration >= -0.5363110089097557
 CBF_Constraint_for_table: x3_input_acceleration >= -0.9141883618941562
 Relaxed_CLF_constraint: - 0.0815393623464684 x1_input_acceleration
   + 0.2395084733695722 x2_input_acceleration
   + 0.1357562953494123 x3_input_acceleration + relaxation_CLF
   <= -20.03297709926187
 CBF_Constraint_obs2: - 0.2890987349138823 x1_input_acceleration
   + 0.1644698075726132 x2_input_acceleration
   + 0.2726158238390198 x3_input_acceleration >= -0.1811847294240089
 CBF_Constraint_obs3: 0.0909012650861177 x1_input_acceleration
   + 0.1644698075726132 x2_input_acceleration
   + 0.2726158238390198 x3_input_acceleration >= -0.0870409312558207
 CBF_Constraint_obs1: - 0.6090987349138823 x1_input_acceleration
   + 0.1644698075726132 x2_input_acceleration
   + 0.2726158238390198 x3_input_acceleration >= -0.5404637173551147
 CBF_Constraint_for_table: x3_input_acceleration >= -0.9204053881116486
 Relaxed_CLF_constraint: - 0.082659652338533 x1_input_acceleration
   + 0.2380084733701994 x2_input_acceleration
   + 0.1370786171425934 x3_input_acceleration + relaxation_CLF
   <= -20.03290847951939
 CBF_Constraint_obs2: - 0.2913833261169815 x1_input_acceleration
   + 0.1611262464047951 x2_input_acceleration
   + 0.2749901007755958 x3_input_acceleration >= -0.1831068319300828
 CBF_Constraint_obs3: 0.0887025045714951 x1_input_acceleration
   + 0.1614698075772805 x2_input_acceleration
   + 0.2752604674442969 x3_input_acceleration >= -0.0871361975174412
 CBF_Constraint_obs1: - 0.611297495428505 x1_input_acceleration
   + 0.1614698075772805 x2_input_acceleration
   + 0.2752604674442969 x3_input_acceleration >= -0.5444068145173249
 CBF_Constraint_for_table: x3_input_acceleration >= -0.9274386999866274
 Relaxed_CLF_constraint: - 0.0837590325958443 x1_input_acceleration
   + 0.236508473372533 x2_input_acceleration
   + 0.138400938945232 x3_input_acceleration + relaxation_CLF
   <= -20.03284266136773
 CBF_Constraint_obs2: - 0.2935405369634196 x1_input_acceleration
   + 0.1581262464048261 x2_input_acceleration
   + 0.2776347443539856 x3_input_acceleration >= -0.1853137598941943
 CBF_Constraint_obs3: 0.086545293725057 x1_input_acceleration
   + 0.1584698075773114 x2_input_acceleration
   + 0.2779051110226868 x3_input_acceleration >= -0.087292523140098
 CBF_Constraint_obs1: - 0.613454706274943 x1_input_acceleration
   + 0.1584698075773114 x2_input_acceleration
   + 0.2779051110226868 x3_input_acceleration >= -0.5483382591212482
 CBF_Constraint_for_table: x3_input_acceleration >= -0.9340503089326019
 Relaxed_CLF_constraint: - 0.0848376380190634 x1_input_acceleration
   + 0.2350084733725485 x2_input_acceleration
   + 0.1397232607344269 x3_input_acceleration + relaxation_CLF
   <= -20.03277959878872
 CBF_Constraint_obs2: - 0.2956564556975636 x1_input_acceleration
   + 0.1551262464170222 x2_input_acceleration
   + 0.2802793880024831 x3_input_acceleration >= -0.1875417843276582
 CBF_Constraint_obs3: 0.0843435443024364 x1_input_acceleration
   + 0.1551262464170222 x2_input_acceleration
   + 0.2802793880024831 x3_input_acceleration >= -0.0871681514149727
 CBF_Constraint_obs1: - 0.6156564556975636 x1_input_acceleration
   + 0.1551262464170222 x2_input_acceleration
   + 0.2802793880024831 x3_input_acceleration >= -0.552066948885709
 CBF_Constraint_for_table: x3_input_acceleration >= -0.9406619180538456
 Relaxed_CLF_constraint: - 0.0858955973861354 x1_input_acceleration
   + 0.2335084733786466 x2_input_acceleration
   + 0.1410455825586757 x3_input_acceleration + relaxation_CLF
   <= -20.03271924685971
 CBF_Constraint_obs2: - 0.2977313290017948 x1_input_acceleration
   + 0.1521262464171385 x2_input_acceleration
   + 0.2829240315813545 x3_input_acceleration >= -0.1897904334554877
 CBF_Constraint_obs3: 0.0822686709982052 x1_input_acceleration
   + 0.1521262464171385 x2_input_acceleration
   + 0.2829240315813545 x3_input_acceleration >= -0.0874456709037827
 CBF_Constraint_obs1: - 0.6177313290017948 x1_input_acceleration
   + 0.1521262464171385 x2_input_acceleration
   + 0.2829240315813545 x3_input_acceleration >= -0.5559754966569236
 CBF_Constraint_for_table: x3_input_acceleration >= -0.9472735270010241
 Relaxed_CLF_constraint: - 0.086933034038251 x1_input_acceleration
   + 0.2320084733787047 x2_input_acceleration
   + 0.1423679043481114 x3_input_acceleration + relaxation_CLF
   <= -20.03266156172603
 CBF_Constraint_obs2: - 0.2997653922770461 x1_input_acceleration
   + 0.1491262464172314 x2_input_acceleration
   + 0.2855686751600996 x3_input_acceleration >= -0.1920592450127483
 CBF_Constraint_obs3: 0.0802346077229539 x1_input_acceleration
   + 0.1491262464172314 x2_input_acceleration
   + 0.2855686751600996 x3_input_acceleration >= -0.0877821223495545
 CBF_Constraint_obs1: - 0.6197653922770461 x1_input_acceleration
   + 0.1491262464172314 x2_input_acceleration
   + 0.2855686751600996 x3_input_acceleration >= -0.5598715588343851
 CBF_Constraint_for_table: x3_input_acceleration >= -0.9535093286625782
 Relaxed_CLF_constraint: - 0.0879500656758767 x1_input_acceleration
   + 0.2305084733787511 x2_input_acceleration
   + 0.1436902261374839 x3_input_acceleration + relaxation_CLF
   <= -20.03260650057569
 CBF_Constraint_obs2: - 0.3016773312805077 x1_input_acceleration
   + 0.1461909770615388 x2_input_acceleration
   + 0.2882503928278297 x3_input_acceleration >= -0.1943366286851683
 CBF_Constraint_obs3: 0.0783226687194923 x1_input_acceleration
   + 0.1461909770615388 x2_input_acceleration
   + 0.2882503928278297 x3_input_acceleration >= -0.088243163968686
 CBF_Constraint_obs1: - 0.6216773312805077 x1_input_acceleration
   + 0.1461909770615388 x2_input_acceleration
   + 0.2882503928278297 x3_input_acceleration >= -0.5636784937095746
 CBF_Constraint_for_table: x3_input_acceleration >= -0.9601209376092776
 Relaxed_CLF_constraint: - 0.0889468047546338 x1_input_acceleration
   + 0.2290084733787918 x2_input_acceleration
   + 0.1450125479268238 x3_input_acceleration + relaxation_CLF
   <= -20.03255402160463
 CBF_Constraint_obs2: - 0.3036504661395518 x1_input_acceleration
   + 0.1434890002855913 x2_input_acceleration
   + 0.2911171233134678 x3_input_acceleration >= -0.196927347845043
 CBF_Constraint_obs3: 0.0763495338604482 x1_input_acceleration
   + 0.1434890002855913 x2_input_acceleration
   + 0.2911171233134678 x3_input_acceleration >= -0.0889594050124687
 CBF_Constraint_obs1: - 0.6236504661395518 x1_input_acceleration
   + 0.1434890002855913 x2_input_acceleration
   + 0.2911171233134678 x3_input_acceleration >= -0.5678477207566843
 CBF_Constraint_for_table: x3_input_acceleration >= -0.9667244999302455
 Relaxed_CLF_constraint: - 0.0899233586038336 x1_input_acceleration
   + 0.2275084733788796 x2_input_acceleration
   + 0.1463348697164263 x3_input_acceleration + relaxation_CLF
   <= -20.03250408399068
 CBF_Constraint_obs2: - 0.3055634081046517 x1_input_acceleration
   + 0.1404890002863931 x2_input_acceleration
   + 0.29376176689619 x3_input_acceleration >= -0.1992526640799644
 CBF_Constraint_obs3: 0.0744365918953483 x1_input_acceleration
   + 0.1404890002863931 x2_input_acceleration
   + 0.29376176689619 x3_input_acceleration >= -0.0894674263805453
 CBF_Constraint_obs1: - 0.6255634081046517 x1_input_acceleration
   + 0.1404890002863931 x2_input_acceleration
   + 0.29376176689619 x3_input_acceleration >= -0.5717033905636858
 CBF_Constraint_for_table: x3_input_acceleration >= -0.9733361088870509
 Relaxed_CLF_constraint: - 0.0908798295863835 x1_input_acceleration
   + 0.2260084733792805 x2_input_acceleration
   + 0.1476571915077874 x3_input_acceleration + relaxation_CLF
   <= -20.03245664786754
 CBF_Constraint_obs2: - 0.3073004808172026 x1_input_acceleration
   + 0.137486020054452 x2_input_acceleration
   + 0.2965115530694459 x3_input_acceleration >= -0.2015688654056899
 CBF_Constraint_obs3: 0.0726995191827974 x1_input_acceleration
   + 0.137486020054452 x2_input_acceleration
   + 0.2965115530694459 x3_input_acceleration >= -0.0901334086293474
 CBF_Constraint_obs1: - 0.6273004808172026 x1_input_acceleration
   + 0.137486020054452 x2_input_acceleration
   + 0.2965115530694459 x3_input_acceleration >= -0.575409250059452
 CBF_Constraint_for_table: x3_input_acceleration >= -0.9799372870238956
 Relaxed_CLF_constraint: - 0.0918163152377029 x1_input_acceleration
   + 0.2245084733794293 x2_input_acceleration
   + 0.1489795132977235 x3_input_acceleration + relaxation_CLF
   <= -20.03241167429817
 CBF_Constraint_obs2: - 0.3091336671884838 x1_input_acceleration
   + 0.1344860200551335 x2_input_acceleration
   + 0.2991561966514362 x3_input_acceleration >= -0.2039306797182785
 CBF_Constraint_obs3: 0.0708663328115162 x1_input_acceleration
   + 0.1344860200551335 x2_input_acceleration
   + 0.2991561966514362 x3_input_acceleration >= -0.0907536958892189
 CBF_Constraint_obs1: - 0.6291336671884838 x1_input_acceleration
   + 0.1344860200551335 x2_input_acceleration
   + 0.2991561966514362 x3_input_acceleration >= -0.5792376134690657
 CBF_Constraint_for_table: x3_input_acceleration >= -0.9865488959788713
 Relaxed_CLF_constraint: - 0.0927329084233435 x1_input_acceleration
   + 0.2230084733797701 x2_input_acceleration
   + 0.1503018350887186 x3_input_acceleration + relaxation_CLF
   <= -20.03236912525395
 CBF_Constraint_obs2: - 0.3109272451708198 x1_input_acceleration
   + 0.1314860200603569 x2_input_acceleration
   + 0.3018008402583606 x3_input_acceleration >= -0.2063100905518502
 CBF_Constraint_obs3: 0.0690727548291802 x1_input_acceleration
   + 0.1314860200603569 x2_input_acceleration
   + 0.3018008402583606 x3_input_acceleration >= -0.0914292076395714
 CBF_Constraint_obs1: - 0.6309272451708198 x1_input_acceleration
   + 0.1314860200603569 x2_input_acceleration
   + 0.3018008402583606 x3_input_acceleration >= -0.5830518866885059
 CBF_Constraint_for_table: x3_input_acceleration >= -0.9931605049961822
 Relaxed_CLF_constraint: - 0.0936296974145115 x1_input_acceleration
   + 0.2215084733823818 x2_input_acceleration
   + 0.1516241568921808 x3_input_acceleration + relaxation_CLF
   <= -20.03232896358856
 CBF_Constraint_obs2: - 0.3127757962718879 x1_input_acceleration
   + 0.1284764833202002 x2_input_acceleration
   + 0.3043651367888979 x3_input_acceleration >= -0.2087163024904546
 CBF_Constraint_obs3: 0.0672242037281121 x1_input_acceleration
   + 0.1284764833202002 x2_input_acceleration
   + 0.3043651367888979 x3_input_acceleration >= -0.0920792960321611
 CBF_Constraint_obs1: - 0.632775796271888 x1_input_acceleration
   + 0.1284764833202002 x2_input_acceleration
   + 0.3043651367888979 x3_input_acceleration >= -0.586936939507965
 CBF_Constraint_for_table: x3_input_acceleration >= -0.9997217480505832
 Relaxed_CLF_constraint: - 0.0945067660863835 x1_input_acceleration
   + 0.2200084733838855 x2_input_acceleration
   + 0.1529464786880281 x3_input_acceleration + relaxation_CLF
   <= -20.03229115301595
 CBF_Constraint_obs2: - 0.3144906527355705 x1_input_acceleration
   + 0.1254764833312438 x2_input_acceleration
   + 0.3070097804201289 x3_input_acceleration >= -0.2111296547510461
 CBF_Constraint_obs3: 0.0655093472644295 x1_input_acceleration
   + 0.1254764833312438 x2_input_acceleration
   + 0.3070097804201289 x3_input_acceleration >= -0.0928635346522541
 CBF_Constraint_obs1: - 0.6344906527355705 x1_input_acceleration
   + 0.1254764833312438 x2_input_acceleration
   + 0.3070097804201289 x3_input_acceleration >= -0.5907221769395024
 CBF_Constraint_for_table: x3_input_acceleration >= -1.006333357128661
 Relaxed_CLF_constraint: - 0.0953641943182248 x1_input_acceleration
   + 0.2185084733894073 x2_input_acceleration
   + 0.1542688005036436 x3_input_acceleration + relaxation_CLF
   <= -20.0322556581239
 CBF_Constraint_obs2: - 0.3161663774016072 x1_input_acceleration
   + 0.1224764833346349 x2_input_acceleration
   + 0.3096544240118992 x3_input_acceleration >= -0.2135594118492696
 CBF_Constraint_obs3: 0.0638336225983928 x1_input_acceleration
   + 0.1224764833346349 x2_input_acceleration
   + 0.3096544240118992 x3_input_acceleration >= -0.0937013533177428
 CBF_Constraint_obs1: - 0.6361663774016072 x1_input_acceleration
   + 0.1224764833346349 x2_input_acceleration
   + 0.3096544240118992 x3_input_acceleration >= -0.5944925137705553
 CBF_Constraint_for_table: x3_input_acceleration >= -1.012944966108087
 Relaxed_CLF_constraint: - 0.0962020566512431 x1_input_acceleration
   + 0.2170084733911028 x2_input_acceleration
   + 0.1555911222995288 x3_input_acceleration + relaxation_CLF
   <= -20.03222244422636
 CBF_Constraint_obs2: - 0.3177144220152527 x1_input_acceleration
   + 0.1194699268237345 x2_input_acceleration
   + 0.3123424597716359 x3_input_acceleration >= -0.2159666619349382
 CBF_Constraint_obs3: 0.0622855779847473 x1_input_acceleration
   + 0.1194699268237345 x2_input_acceleration
   + 0.3123424597716359 x3_input_acceleration >= -0.0946379610204481
 CBF_Constraint_obs1: - 0.6377144220152527 x1_input_acceleration
   + 0.1194699268237345 x2_input_acceleration
   + 0.3123424597716359 x3_input_acceleration >= -0.5981381995471403
 CBF_Constraint_for_table: x3_input_acceleration >= -1.019946091407544
 Relaxed_CLF_constraint: - 0.0970204248137788 x1_input_acceleration
   + 0.2155084733911152 x2_input_acceleration
   + 0.1569134440886988 x3_input_acceleration + relaxation_CLF
   <= -20.03219147754808
 CBF_Constraint_obs2: - 0.3193123040818462 x1_input_acceleration
   + 0.1164699268245346 x2_input_acceleration
   + 0.3149871033539759 x3_input_acceleration >= -0.2184280833398727
 CBF_Constraint_obs3: 0.0606876959181538 x1_input_acceleration
   + 0.1164699268245346 x2_input_acceleration
   + 0.3149871033539759 x3_input_acceleration >= -0.0955813944621188
 CBF_Constraint_obs1: - 0.6393123040818462 x1_input_acceleration
   + 0.1164699268245346 x2_input_acceleration
   + 0.3149871033539759 x3_input_acceleration >= -0.6018779266053498
 CBF_Constraint_for_table: x3_input_acceleration >= -1.026557700363394
 Relaxed_CLF_constraint: - 0.0978193658470755 x1_input_acceleration
   + 0.2140084733915153 x2_input_acceleration
   + 0.1582357658798688 x3_input_acceleration + relaxation_CLF
   <= -20.03216272504867
 CBF_Constraint_obs2: - 0.321078882533095 x1_input_acceleration
   + 0.1131289882685528 x2_input_acceleration
   + 0.3172865168920215 x3_input_acceleration >= -0.220700688246048
 CBF_Constraint_obs3: 0.058921117466905 x1_input_acceleration
   + 0.1131289882685528 x2_input_acceleration
   + 0.3172865168920215 x3_input_acceleration >= -0.0961757498396078
 CBF_Constraint_obs1: - 0.6410788825330951 x1_input_acceleration
   + 0.1131289882685528 x2_input_acceleration
   + 0.3172865168920215 x3_input_acceleration >= -0.6055637942725243
 CBF_Constraint_for_table: x3_input_acceleration >= -1.033269147244855
 Relaxed_CLF_constraint: - 0.0985989429907907 x1_input_acceleration
   + 0.212508473397582 x2_input_acceleration
   + 0.1595580877001612 x3_input_acceleration + relaxation_CLF
   <= -20.03213615447008
 CBF_Constraint_obs2: - 0.3225994276911506 x1_input_acceleration
   + 0.1101289882689043 x2_input_acceleration
   + 0.3199311604720425 x3_input_acceleration >= -0.2231934153059727
 CBF_Constraint_obs3: 0.0574005723088494 x1_input_acceleration
   + 0.1101289882689043 x2_input_acceleration
   + 0.3199311604720425 x3_input_acceleration >= -0.0972239589993796
 CBF_Constraint_obs1: - 0.6425994276911506 x1_input_acceleration
   + 0.1101289882689043 x2_input_acceleration
   + 0.3199311604720425 x3_input_acceleration >= -0.6092729574588933
 CBF_Constraint_for_table: x3_input_acceleration >= -1.039880756194907
 Relaxed_CLF_constraint: - 0.0993592155698185 x1_input_acceleration
   + 0.2110084733977577 x2_input_acceleration
   + 0.1608804094901717 x3_input_acceleration + relaxation_CLF
   <= -20.03211173428881
 CBF_Constraint_obs2: - 0.3240814747820489 x1_input_acceleration
   + 0.1071289882689487 x2_input_acceleration
   + 0.3225758040504855 x3_input_acceleration >= -0.2257007147290669
 CBF_Constraint_obs3: 0.0559185252179511 x1_input_acceleration
   + 0.1071289882689487 x2_input_acceleration
   + 0.3225758040504855 x3_input_acceleration >= -0.0983233136861204
 CBF_Constraint_obs1: - 0.6440814747820489 x1_input_acceleration
   + 0.1071289882689487 x2_input_acceleration
   + 0.3225758040504855 x3_input_acceleration >= -0.6129658945547061
 CBF_Constraint_for_table: x3_input_acceleration >= -1.046492365141015
 Relaxed_CLF_constraint: - 0.1001002391152677 x1_input_acceleration
   + 0.2095084733977799 x2_input_acceleration
   + 0.1622027312793932 x3_input_acceleration + relaxation_CLF
   <= -20.03208943373236
 CBF_Constraint_obs2: - 0.3254989013317586 x1_input_acceleration
   + 0.1041245775257459 x2_input_acceleration
   + 0.3252340374902567 x3_input_acceleration >= -0.2282108019441384
 CBF_Constraint_obs3: 0.0545010986682413 x1_input_acceleration
   + 0.1041245775257459 x2_input_acceleration
   + 0.3252340374902567 x3_input_acceleration >= -0.0994868456789677
 CBF_Constraint_obs1: - 0.6454989013317587 x1_input_acceleration
   + 0.1041245775257459 x2_input_acceleration
   + 0.3252340374902567 x3_input_acceleration >= -0.6166099230095454
 CBF_Constraint_for_table: x3_input_acceleration >= -1.052628627050837
 Relaxed_CLF_constraint: - 0.1008220654119732 x1_input_acceleration
   + 0.2080084733980352 x2_input_acceleration
   + 0.1635250530697744 x3_input_acceleration + relaxation_CLF
   <= -20.03206922274432
 CBF_Constraint_obs2: - 0.3269042557351064 x1_input_acceleration
   + 0.1011245775259143 x2_input_acceleration
   + 0.3278786810692998 x3_input_acceleration >= -0.2307462526215777
 CBF_Constraint_obs3: 0.0530957442648936 x1_input_acceleration
   + 0.1011245775259143 x2_input_acceleration
   + 0.3278786810692998 x3_input_acceleration >= -0.1006872096732266
 CBF_Constraint_obs1: - 0.6469042557351063 x1_input_acceleration
   + 0.1011245775259143 x2_input_acceleration
   + 0.3278786810692998 x3_input_acceleration >= -0.6202696572096628
 CBF_Constraint_for_table: x3_input_acceleration >= -1.059240235998445
 Relaxed_CLF_constraint: - 0.101524742613647 x1_input_acceleration
   + 0.2065084733981195 x2_input_acceleration
   + 0.164847374859296 x3_input_acceleration + relaxation_CLF
   <= -20.0320510719784
 CBF_Constraint_obs2: - 0.3282714009191263 x1_input_acceleration
   + 0.0981245775260086 x2_input_acceleration
   + 0.3305233246479484 x3_input_acceleration >= -0.2332952668904956
 CBF_Constraint_obs3: 0.0517285990808737 x1_input_acceleration
   + 0.0981245775260086 x2_input_acceleration
   + 0.3305233246479484 x3_input_acceleration >= -0.1019374360173256
 CBF_Constraint_obs1: - 0.6482714009191264 x1_input_acceleration
   + 0.0981245775260086 x2_input_acceleration
   + 0.3305233246479484 x3_input_acceleration >= -0.6239123876257968
 CBF_Constraint_for_table: x3_input_acceleration >= -1.065851844945066
 Relaxed_CLF_constraint: - 0.102208315205657 x1_input_acceleration
   + 0.2050084733981666 x2_input_acceleration
   + 0.1661696966486202 x3_input_acceleration + relaxation_CLF
   <= -20.03203495277861
 CBF_Constraint_obs2: - 0.3296123396889084 x1_input_acceleration
   + 0.0951163520859062 x2_input_acceleration
   + 0.3331528286504758 x3_input_acceleration >= -0.2358527776839245
 CBF_Constraint_obs3: 0.0503876603110917 x1_input_acceleration
   + 0.0951163520859062 x2_input_acceleration
   + 0.3331528286504758 x3_input_acceleration >= -0.1032210549794616
 CBF_Constraint_obs1: - 0.6496123396889083 x1_input_acceleration
   + 0.0951163520859062 x2_input_acceleration
   + 0.3331528286504758 x3_input_acceleration >= -0.6275426494350511
 CBF_Constraint_for_table: x3_input_acceleration >= -1.072869659554961
 Relaxed_CLF_constraint: - 0.1028728241260704 x1_input_acceleration
   + 0.2035084733986049 x2_input_acceleration
   + 0.1674920184397704 x3_input_acceleration + relaxation_CLF
   <= -20.03202083717189
 CBF_Constraint_obs2: - 0.3308117522575353 x1_input_acceleration
   + 0.0921708307314054 x2_input_acceleration
   + 0.3358288242726987 x3_input_acceleration >= -0.2383910958522518
 CBF_Constraint_obs3: 0.0491882477424647 x1_input_acceleration
   + 0.0921708307314054 x2_input_acceleration
   + 0.3358288242726987 x3_input_acceleration >= -0.1046199312075933
 CBF_Constraint_obs1: - 0.6509033049919103 x1_input_acceleration
   + 0.0921163520860807 x2_input_acceleration
   + 0.3357974722295469 x3_input_acceleration >= -0.6311505964084897
 CBF_Constraint_for_table: x3_input_acceleration >= -1.079074466802047
 Relaxed_CLF_constraint: - 0.1035183067775714 x1_input_acceleration
   + 0.2020084733986922 x2_input_acceleration
   + 0.1688143402293059 x3_input_acceleration + relaxation_CLF
   <= -20.03200869785201
 CBF_Constraint_obs2: - 0.3320647328506212 x1_input_acceleration
   + 0.0891708307324683 x2_input_acceleration
   + 0.3384734678562258 x3_input_acceleration >= -0.2409783903741156
 CBF_Constraint_obs3: 0.0479352671493788 x1_input_acceleration
   + 0.0891708307324683 x2_input_acceleration
   + 0.3384734678562258 x3_input_acceleration >= -0.1060168941660255
 CBF_Constraint_obs1: - 0.6520647328506212 x1_input_acceleration
   + 0.0891708307324683 x2_input_acceleration
   + 0.3384734678562258 x3_input_acceleration >= -0.6346301766546125
 CBF_Constraint_for_table: x3_input_acceleration >= -1.085686075760865
 Relaxed_CLF_constraint: - 0.1041447970741144 x1_input_acceleration
   + 0.2005084733992236 x2_input_acceleration
   + 0.1701366620210695 x3_input_acceleration + relaxation_CLF
   <= -20.03199850817047
 CBF_Constraint_obs2: - 0.3332797897632375 x1_input_acceleration
   + 0.0861708307390515 x2_input_acceleration
   + 0.3411181114670633 x3_input_acceleration >= -0.2435779953821773
 CBF_Constraint_obs3: 0.0467202102367626 x1_input_acceleration
   + 0.0861708307390515 x2_input_acceleration
   + 0.3411181114670633 x3_input_acceleration >= -0.1074621951071017
 CBF_Constraint_obs1: - 0.6532797897632374 x1_input_acceleration
   + 0.0861708307390515 x2_input_acceleration
   + 0.3411181114670633 x3_input_acceleration >= -0.6382018271927672
 CBF_Constraint_for_table: x3_input_acceleration >= -1.092297684787959
 Relaxed_CLF_constraint: - 0.1047523255304225 x1_input_acceleration
   + 0.1990084734025152 x2_input_acceleration
   + 0.1714589838264882 x3_input_acceleration + relaxation_CLF
   <= -20.03199024212994
 CBF_Constraint_obs2: - 0.3344288436403479 x1_input_acceleration
   + 0.0834372635014664 x2_input_acceleration
   + 0.3440707918507912 x3_input_acceleration >= -0.2464864227550798
 CBF_Constraint_obs3: 0.0455711563596521 x1_input_acceleration
   + 0.0834372635014664 x2_input_acceleration
   + 0.3440707918507912 x3_input_acceleration >= -0.1092790212967493
 CBF_Constraint_obs1: - 0.6544288436403479 x1_input_acceleration
   + 0.0834372635014664 x2_input_acceleration
   + 0.3440707918507912 x3_input_acceleration >= -0.6420294976673581
 CBF_Constraint_for_table: x3_input_acceleration >= -1.099320267762507
 Relaxed_CLF_constraint: - 0.1053409191651447 x1_input_acceleration
   + 0.1975084734026497 x2_input_acceleration
   + 0.1727813056162525 x3_input_acceleration + relaxation_CLF
   <= -20.03198387435474
 CBF_Constraint_obs2: - 0.3355682086624776 x1_input_acceleration
   + 0.08043726350241 x2_input_acceleration
   + 0.3467154354333056 x3_input_acceleration >= -0.2491097114905086
 CBF_Constraint_obs3: 0.0444317913375224 x1_input_acceleration
   + 0.08043726350241 x2_input_acceleration
   + 0.3467154354333056 x3_input_acceleration >= -0.1108199132611549
 CBF_Constraint_obs1: - 0.6555682086624777 x1_input_acceleration
   + 0.08043726350241 x2_input_acceleration
   + 0.3467154354333056 x3_input_acceleration >= -0.6455642784204908
 CBF_Constraint_for_table: x3_input_acceleration >= -1.105931876718793
 Relaxed_CLF_constraint: - 0.1059106016762096 x1_input_acceleration
   + 0.1960084734031214 x2_input_acceleration
   + 0.1741036274075097 x3_input_acceleration + relaxation_CLF
   <= -20.03197938010788
 CBF_Constraint_obs2: - 0.3366697920010148 x1_input_acceleration
   + 0.0774372635025409 x2_input_acceleration
   + 0.3493600790121303 x3_input_acceleration >= -0.2517444292901631
 CBF_Constraint_obs3: 0.0433302079989852 x1_input_acceleration
   + 0.0774372635025409 x2_input_acceleration
   + 0.3493600790121303 x3_input_acceleration >= -0.112408126889199
 CBF_Constraint_obs1: - 0.6566697920010148 x1_input_acceleration
   + 0.0774372635025409 x2_input_acceleration
   + 0.3493600790121303 x3_input_acceleration >= -0.6490802628909749
 CBF_Constraint_for_table: x3_input_acceleration >= -1.112543485665854
 Relaxed_CLF_constraint: - 0.1064613933454782 x1_input_acceleration
   + 0.1945084734031869 x2_input_acceleration
   + 0.1754259491969221 x3_input_acceleration + relaxation_CLF
   <= -20.03197673526014
 CBF_Constraint_obs2: - 0.337678314504575 x1_input_acceleration
   + 0.0745148687504793 x2_input_acceleration
   + 0.3519830265018101 x3_input_acceleration >= -0.2543389508711572
 CBF_Constraint_obs3: 0.0423216854954251 x1_input_acceleration
   + 0.0745148687504793 x2_input_acceleration
   + 0.3519830265018101 x3_input_acceleration >= -0.1140445520918109
 CBF_Constraint_obs1: - 0.6576783145045749 x1_input_acceleration
   + 0.0745148687504793 x2_input_acceleration
   + 0.3519830265018101 x3_input_acceleration >= -0.652481602474817
 CBF_Constraint_for_table: x3_input_acceleration >= -1.118781671517282
 Relaxed_CLF_constraint: - 0.106993311152434 x1_input_acceleration
   + 0.1930084734034073 x2_input_acceleration
   + 0.1767482709871111 x3_input_acceleration + relaxation_CLF
   <= -20.03197591629352
 CBF_Constraint_obs2: - 0.3388021811224688 x1_input_acceleration
   + 0.0711786985585474 x2_input_acceleration
   + 0.3543552768756053 x3_input_acceleration >= -0.256776234139272
 CBF_Constraint_obs3: 0.0411978188775312 x1_input_acceleration
   + 0.0711786985585474 x2_input_acceleration
   + 0.3543552768756053 x3_input_acceleration >= -0.1154141620729266
 CBF_Constraint_obs1: - 0.6587044295050372 x1_input_acceleration
   + 0.0715148687550806 x2_input_acceleration
   + 0.3546276701022288 x3_input_acceleration >= -0.6559583578809349
 CBF_Constraint_for_table: x3_input_acceleration >= -1.125387916100299
 Relaxed_CLF_constraint: - 0.1075063686526652 x1_input_acceleration
   + 0.191508473405708 x2_input_acceleration
   + 0.1780705927873205 x3_input_acceleration + relaxation_CLF
   <= -20.03197690028108
 CBF_Constraint_obs2: - 0.3397905964204775 x1_input_acceleration
   + 0.0681786985593851 x2_input_acceleration
   + 0.3569999204577541 x3_input_acceleration >= -0.2594436594498476
 CBF_Constraint_obs3: 0.0402094035795225 x1_input_acceleration
   + 0.0681786985593851 x2_input_acceleration
   + 0.3569999204577541 x3_input_acceleration >= -0.117142592850394
 CBF_Constraint_obs1: - 0.6597905964204775 x1_input_acceleration
   + 0.0681786985593851 x2_input_acceleration
   + 0.3569999204577541 x3_input_acceleration >= -0.6592761365862296
 CBF_Constraint_for_table: x3_input_acceleration >= -1.131999525055671
 Relaxed_CLF_constraint: - 0.1080005763016695 x1_input_acceleration
   + 0.1900084734061268 x2_input_acceleration
   + 0.1793929145783949 x3_input_acceleration + relaxation_CLF
   <= -20.03197966489942
 CBF_Constraint_obs2: - 0.3407413255936279 x1_input_acceleration
   + 0.0651786985761822 x2_input_acceleration
   + 0.3596445641184245 x3_input_acceleration >= -0.2621214077691489
 CBF_Constraint_obs3: 0.0392586744063721 x1_input_acceleration
   + 0.0651786985761822 x2_input_acceleration
   + 0.3596445641184245 x3_input_acceleration >= -0.1189171484552024
 CBF_Constraint_obs1: - 0.6607413255936279 x1_input_acceleration
   + 0.0651786985761822 x2_input_acceleration
   + 0.3596445641184245 x3_input_acceleration >= -0.6627144682440511
 CBF_Constraint_for_table: x3_input_acceleration >= -1.138611134207347
 Relaxed_CLF_constraint: - 0.1084759408882447 x1_input_acceleration
   + 0.1885084734145254 x2_input_acceleration
   + 0.1807152364087301 x3_input_acceleration + relaxation_CLF
   <= -20.03198418838837
 CBF_Constraint_obs2: - 0.3417173190738104 x1_input_acceleration
   + 0.0621251736195551 x2_input_acceleration
   + 0.3622615512086734 x3_input_acceleration >= -0.2648296185454581
 CBF_Constraint_obs3: 0.0382826809261896 x1_input_acceleration
   + 0.0621251736195551 x2_input_acceleration
   + 0.3622615512086734 x3_input_acceleration >= -0.1206981654253382
 CBF_Constraint_obs1: - 0.6617173190738105 x1_input_acceleration
   + 0.0621251736195551 x2_input_acceleration
   + 0.3622615512086734 x3_input_acceleration >= -0.6662034738045065
 CBF_Constraint_for_table: x3_input_acceleration >= -1.145224233437028
 Relaxed_CLF_constraint: - 0.1089324663758945 x1_input_acceleration
   + 0.1870084734217159 x2_input_acceleration
   + 0.1820375582314424 x3_input_acceleration + relaxation_CLF
   <= -20.03199044958765
 CBF_Constraint_obs2: - 0.3425926925131664 x1_input_acceleration
   + 0.0591251736206971 x2_input_acceleration
   + 0.3649061947917515 x3_input_acceleration >= -0.2675275876456178
 CBF_Constraint_obs3: 0.0374073074868336 x1_input_acceleration
   + 0.0591251736206971 x2_input_acceleration
   + 0.3649061947917515 x3_input_acceleration >= -0.1225645297581097
 CBF_Constraint_obs1: - 0.6625926925131664 x1_input_acceleration
   + 0.0591251736206971 x2_input_acceleration
   + 0.3649061947917515 x3_input_acceleration >= -0.6696017416561508
 CBF_Constraint_for_table: x3_input_acceleration >= -1.151835842394723
 Relaxed_CLF_constraint: - 0.1093701530955725 x1_input_acceleration
   + 0.185508473422287 x2_input_acceleration
   + 0.1833598800229815 x3_input_acceleration + relaxation_CLF
   <= -20.03199842788066
 CBF_Constraint_obs2: - 0.3434303825576066 x1_input_acceleration
   + 0.0561251736248739 x2_input_acceleration
   + 0.3675508383808149 x3_input_acceleration >= -0.2702351019656577
 CBF_Constraint_obs3: 0.0365696174423934 x1_input_acceleration
   + 0.0561251736248739 x2_input_acceleration
   + 0.3675508383808149 x3_input_acceleration >= -0.1244762385359314
 CBF_Constraint_obs1: - 0.6634303825576066 x1_input_acceleration
   + 0.0561251736248739 x2_input_acceleration
   + 0.3675508383808149 x3_input_acceleration >= -0.6729794080117429
 CBF_Constraint_for_table: x3_input_acceleration >= -1.158447451367381
 Relaxed_CLF_constraint: - 0.1097889981177926 x1_input_acceleration
   + 0.1840084734243754 x2_input_acceleration
   + 0.1846822018175132 x3_input_acceleration + relaxation_CLF
   <= -20.03200810322714
 CBF_Constraint_obs2: - 0.3441435923298702 x1_input_acceleration
   + 0.053452760756741 x2_input_acceleration
   + 0.3705137707810744 x3_input_acceleration >= -0.2732155801407948
 CBF_Constraint_obs3: 0.0358564076701298 x1_input_acceleration
   + 0.053452760756741 x2_input_acceleration
   + 0.3705137707810744 x3_input_acceleration >= -0.1267791674274181
 CBF_Constraint_obs1: - 0.6641435923298702 x1_input_acceleration
   + 0.053452760756741 x2_input_acceleration
   + 0.3705137707810744 x3_input_acceleration >= -0.676530454004691
 CBF_Constraint_for_table: x3_input_acceleration >= -1.165065616871204
 Relaxed_CLF_constraint: - 0.1101889951853209 x1_input_acceleration
   + 0.1825084734264661 x2_input_acceleration
   + 0.1860045236160927 x3_input_acceleration + relaxation_CLF
   <= -20.03201945613522
 CBF_Constraint_obs2: - 0.3449297134042108 x1_input_acceleration
   + 0.0504912653575205 x2_input_acceleration
   + 0.3731144261325119 x3_input_acceleration >= -0.2759253125683853
 CBF_Constraint_obs3: 0.0350702865957892 x1_input_acceleration
   + 0.0504912653575205 x2_input_acceleration
   + 0.3731144261325119 x3_input_acceleration >= -0.1287420848343851
 CBF_Constraint_obs1: - 0.6649297134042108 x1_input_acceleration
   + 0.0504912653575205 x2_input_acceleration
   + 0.3731144261325119 x3_input_acceleration >= -0.679869083291754
 CBF_Constraint_for_table: x3_input_acceleration >= -1.171648019543469
 Relaxed_CLF_constraint: - 0.1105701347935361 x1_input_acceleration
   + 0.1810084734265934 x2_input_acceleration
   + 0.1873268454057335 x3_input_acceleration + relaxation_CLF
   <= -20.03203246766806
 CBF_Constraint_obs2: - 0.3456542518962804 x1_input_acceleration
   + 0.0474912653578043 x2_input_acceleration
   + 0.3757590697119371 x3_input_acceleration >= -0.2786589507625538
 CBF_Constraint_obs3: 0.0343457481037196 x1_input_acceleration
   + 0.0474912653578043 x2_input_acceleration
   + 0.3757590697119371 x3_input_acceleration >= -0.1307874114610874
 CBF_Constraint_obs1: - 0.6656542518962804 x1_input_acceleration
   + 0.0474912653578043 x2_input_acceleration
   + 0.3757590697119371 x3_input_acceleration >= -0.6831823522795781
 CBF_Constraint_for_table: x3_input_acceleration >= -1.178259628492031
 Relaxed_CLF_constraint: - 0.1109324040395709 x1_input_acceleration
   + 0.1795084734267354 x2_input_acceleration
   + 0.1886491671954461 x3_input_acceleration + relaxation_CLF
   <= -20.03204711943261
 CBF_Constraint_obs2: - 0.3463410160841648 x1_input_acceleration
   + 0.0444912653843449 x2_input_acceleration
   + 0.3784037133955739 x3_input_acceleration >= -0.2814011780365894
 CBF_Constraint_obs3: 0.0336589839158352 x1_input_acceleration
   + 0.0444912653843449 x2_input_acceleration
   + 0.3784037133955739 x3_input_acceleration >= -0.1328772127566329
 CBF_Constraint_obs1: - 0.6663410160841647 x1_input_acceleration
   + 0.0444912653843449 x2_input_acceleration
   + 0.3784037133955739 x3_input_acceleration >= -0.686473990903921
 CBF_Constraint_for_table: x3_input_acceleration >= -1.184871237701124
 Relaxed_CLF_constraint: - 0.1112757861335131 x1_input_acceleration
   + 0.1780084734400056 x2_input_acceleration
   + 0.1899714890372645 x3_input_acceleration + relaxation_CLF
   <= -20.03206339353724
 CBF_Constraint_obs2: - 0.346989968993076 x1_input_acceleration
   + 0.0414912653843985 x2_input_acceleration
   + 0.3810483569740293 x3_input_acceleration >= -0.2841517675470272
 CBF_Constraint_obs3: 0.033010031006924 x1_input_acceleration
   + 0.0414912653843985 x2_input_acceleration
   + 0.3810483569740293 x3_input_acceleration >= -0.135011297003605
 CBF_Constraint_obs1: - 0.6669899689930761 x1_input_acceleration
   + 0.0414912653843985 x2_input_acceleration
   + 0.3810483569740293 x3_input_acceleration >= -0.6897437427414879
 CBF_Constraint_for_table: x3_input_acceleration >= -1.19146347513771
 Relaxed_CLF_constraint: - 0.1116002625879688 x1_input_acceleration
   + 0.1765084734400325 x2_input_acceleration
   + 0.1912938108264922 x3_input_acceleration + relaxation_CLF
   <= -20.03208127274654
 CBF_Constraint_obs2: - 0.3475128485957881 x1_input_acceleration
   + 0.0385403796117431 x2_input_acceleration
   + 0.3837483136627983 x3_input_acceleration >= -0.2868916362985223
 CBF_Constraint_obs3: 0.0324871514042119 x1_input_acceleration
   + 0.0385403796117431 x2_input_acceleration
   + 0.3837483136627983 x3_input_acceleration >= -0.1372544301325236
 CBF_Constraint_obs1: - 0.6675128485957882 x1_input_acceleration
   + 0.0385403796117431 x2_input_acceleration
   + 0.3837483136627983 x3_input_acceleration >= -0.6929019151751528
 CBF_Constraint_for_table: x3_input_acceleration >= -1.198075084083754
 Relaxed_CLF_constraint: - 0.1119058098264586 x1_input_acceleration
   + 0.1750084734400573 x2_input_acceleration
   + 0.1926161326157009 x3_input_acceleration + relaxation_CLF
   <= -20.03210074023703
 CBF_Constraint_obs2: - 0.3481766307898824 x1_input_acceleration
   + 0.0354893580428867 x2_input_acceleration
   + 0.3863480153688455 x3_input_acceleration >= -0.2896890621774199
 CBF_Constraint_obs3: 0.0319139682701762 x1_input_acceleration
   + 0.0355403796188144 x2_input_acceleration
   + 0.3863929572710061 x3_input_acceleration >= -0.139476421728541
 CBF_Constraint_obs1: - 0.6680860317298238 x1_input_acceleration
   + 0.0355403796188144 x2_input_acceleration
   + 0.3863929572710061 x3_input_acceleration >= -0.6961269772557327
 CBF_Constraint_for_table: x3_input_acceleration >= -1.204685202988154
 Relaxed_CLF_constraint: - 0.1121924013934764 x1_input_acceleration
   + 0.1735084734435929 x2_input_acceleration
   + 0.1939384544198048 x3_input_acceleration + relaxation_CLF
   <= -20.03212177975596
 CBF_Constraint_obs2: - 0.3487118427743366 x1_input_acceleration
   + 0.0324893580532518 x2_input_acceleration
   + 0.3889926589906518 x3_input_acceleration >= -0.2924634955330044
 CBF_Constraint_obs3: 0.0312881572256634 x1_input_acceleration
   + 0.0324893580532518 x2_input_acceleration
   + 0.3889926589906518 x3_input_acceleration >= -0.1416872448973847
 CBF_Constraint_obs1: - 0.6687118427743366 x1_input_acceleration
   + 0.0324893580532518 x2_input_acceleration
   + 0.3889926589906518 x3_input_acceleration >= -0.6994329697524737
 CBF_Constraint_for_table: x3_input_acceleration >= -1.211296812042669
 Relaxed_CLF_constraint: - 0.1124600073857035 x1_input_acceleration
   + 0.1720084734487755 x2_input_acceleration
   + 0.195260776230708 x3_input_acceleration + relaxation_CLF
   <= -20.03214437557343
 CBF_Constraint_obs2: - 0.3491856519108181 x1_input_acceleration
   + 0.0294804173566396 x2_input_acceleration
   + 0.391615964106483 x3_input_acceleration >= -0.2952034723136936
 CBF_Constraint_obs3: 0.0307909830684299 x1_input_acceleration
   + 0.029489358053356 x2_input_acceleration
   + 0.3916373025693126 x3_input_acceleration >= -0.1439968545506083
 CBF_Constraint_obs1: - 0.6692090169315701 x1_input_acceleration
   + 0.029489358053356 x2_input_acceleration
   + 0.3916373025693126 x3_input_acceleration >= -0.7026126341808558
 CBF_Constraint_for_table: x3_input_acceleration >= -1.218329229781436
 Relaxed_CLF_constraint: - 0.1127085944643203 x1_input_acceleration
   + 0.1705084734488276 x2_input_acceleration
   + 0.1965830980200384 x3_input_acceleration + relaxation_CLF
   <= -20.03216851248445
 CBF_Constraint_obs2: - 0.3496366080497486 x1_input_acceleration
   + 0.0261256505115279 x2_input_acceleration
   + 0.3940558061276484 x3_input_acceleration >= -0.2977603570576185
 CBF_Constraint_obs3: 0.0303633919502515 x1_input_acceleration
   + 0.0261256505115279 x2_input_acceleration
   + 0.3940558061276484 x3_input_acceleration >= -0.1461055794103574
 CBF_Constraint_obs1: - 0.669644714281438 x1_input_acceleration
   + 0.0264804173572311 x2_input_acceleration
   + 0.3942606076870966 x3_input_acceleration >= -0.7057083533028508
 CBF_Constraint_for_table: x3_input_acceleration >= -1.224981369891417
 Relaxed_CLF_constraint: - 0.1129381256496302 x1_input_acceleration
   + 0.1690084734491233 x2_input_acceleration
   + 0.1979054198103452 x3_input_acceleration + relaxation_CLF
   <= -20.03219417580527
 CBF_Constraint_obs2: - 0.3500574776497797 x1_input_acceleration
   + 0.023125650511555 x2_input_acceleration
   + 0.396700449705973 x3_input_acceleration >= -0.3005578502087872
 CBF_Constraint_obs3: 0.0299425223502203 x1_input_acceleration
   + 0.023125650511555 x2_input_acceleration
   + 0.396700449705973 x3_input_acceleration >= -0.1485032464414965
 CBF_Constraint_obs1: - 0.6700574776497797 x1_input_acceleration
   + 0.023125650511555 x2_input_acceleration
   + 0.396700449705973 x3_input_acceleration >= -0.708603832328611
 CBF_Constraint_for_table: x3_input_acceleration >= -1.231592978837229
 Relaxed_CLF_constraint: - 0.1131485604496458 x1_input_acceleration
   + 0.1675084734491368 x2_input_acceleration
   + 0.1992277415995075 x3_input_acceleration + relaxation_CLF
   <= -20.03222135137277
 CBF_Constraint_obs2: - 0.3504400661878954 x1_input_acceleration
   + 0.0201256505162322 x2_input_acceleration
   + 0.3993450933029651 x3_input_acceleration >= -0.3033622316795399
 CBF_Constraint_obs3: 0.0295599338121046 x1_input_acceleration
   + 0.0201256505162322 x2_input_acceleration
   + 0.3993450933029651 x3_input_acceleration >= -0.1509441688010393
 CBF_Constraint_obs1: - 0.6704400661878953 x1_input_acceleration
   + 0.0201256505162322 x2_input_acceleration
   + 0.3993450933029651 x3_input_acceleration >= -0.7117142846298561
 CBF_Constraint_for_table: x3_input_acceleration >= -1.238204587829709
 Relaxed_CLF_constraint: - 0.1133398547187036 x1_input_acceleration
   + 0.1660084734514754 x2_input_acceleration
   + 0.2005500633980036 x3_input_acceleration + relaxation_CLF
   <= -20.03225002554171
 CBF_Constraint_obs2: - 0.3507842779827595 x1_input_acceleration
   + 0.0171256505168002 x2_input_acceleration
   + 0.4019897368834532 x3_input_acceleration >= -0.3061733076814218
 CBF_Constraint_obs3: 0.0292157220172405 x1_input_acceleration
   + 0.0171256505168002 x2_input_acceleration
   + 0.4019897368834532 x3_input_acceleration >= -0.1534282435978003
 CBF_Constraint_obs1: - 0.6707842779827595 x1_input_acceleration
   + 0.0171256505168002 x2_input_acceleration
   + 0.4019897368834532 x3_input_acceleration >= -0.7148007300676295
 CBF_Constraint_for_table: x3_input_acceleration >= -1.244425488334427
 Relaxed_CLF_constraint: - 0.1135119606161357 x1_input_acceleration
   + 0.1645084734517594 x2_input_acceleration
   + 0.2018723851882476 x3_input_acceleration + relaxation_CLF
   <= -20.03228018517678
 CBF_Constraint_obs2: - 0.3510857180459558 x1_input_acceleration
   + 0.0141162329883034 x2_input_acceleration
   + 0.4046440364328542 x3_input_acceleration >= -0.3089965570878719
 CBF_Constraint_obs3: 0.0289142819540442 x1_input_acceleration
   + 0.0141162329883034 x2_input_acceleration
   + 0.4046440364328542 x3_input_acceleration >= -0.1559651249442139
 CBF_Constraint_obs1: - 0.6711701182229577 x1_input_acceleration
   + 0.0141214781970437 x2_input_acceleration
   + 0.4045713187662283 x3_input_acceleration >= -0.7179333696376671
 CBF_Constraint_for_table: x3_input_acceleration >= -1.25141260658888
 Relaxed_CLF_constraint: - 0.1136648264149457 x1_input_acceleration
   + 0.1630084734544483 x2_input_acceleration
   + 0.2031947069867213 x3_input_acceleration + relaxation_CLF
   <= -20.03231181765172
 CBF_Constraint_obs2: - 0.3513528593376736 x1_input_acceleration
   + 0.0111162330071388 x2_input_acceleration
   + 0.4072886800633644 x3_input_acceleration >= -0.311820589135982
 CBF_Constraint_obs3: 0.0286471406623265 x1_input_acceleration
   + 0.0111162330071388 x2_input_acceleration
   + 0.4072886800633644 x3_input_acceleration >= -0.1585353727651922
 CBF_Constraint_obs1: - 0.6713528593376735 x1_input_acceleration
   + 0.0111162330071388 x2_input_acceleration
   + 0.4072886800633644 x3_input_acceleration >= -0.7209028766061208
 CBF_Constraint_for_table: x3_input_acceleration >= -1.258024215665156
 Relaxed_CLF_constraint: - 0.1137983970608046 x1_input_acceleration
   + 0.161508473463866 x2_input_acceleration
   + 0.2045170288019764 x3_input_acceleration + relaxation_CLF
   <= -20.03234491089769
 CBF_Constraint_obs2: - 0.3515812918698246 x1_input_acceleration
   + 0.00811623301061759 x2_input_acceleration
   + 0.409933323652169 x3_input_acceleration >= -0.3146507598395643
 CBF_Constraint_obs3: 0.0284187081301754 x1_input_acceleration
   + 0.00811623301061759 x2_input_acceleration
   + 0.409933323652169 x3_input_acceleration >= -0.1611485325632308
 CBF_Constraint_obs1: - 0.6715812918698246 x1_input_acceleration
   + 0.00811623301061759 x2_input_acceleration
   + 0.409933323652169 x3_input_acceleration >= -0.7239157933354239
 CBF_Constraint_for_table: x3_input_acceleration >= -1.264635824637167
 Relaxed_CLF_constraint: - 0.1139126133268802 x1_input_acceleration
   + 0.1600084734656054 x2_input_acceleration
   + 0.2058393505963787 x3_input_acceleration + relaxation_CLF
   <= -20.03237945332388
 CBF_Constraint_obs2: - 0.3517131914771827 x1_input_acceleration
   + 0.00553263107913449 x2_input_acceleration
   + 0.4128296180961009 x3_input_acceleration >= -0.3177013408038695
 CBF_Constraint_obs3: 0.0282868085228174 x1_input_acceleration
   + 0.00553263107913449 x2_input_acceleration
   + 0.4128296180961009 x3_input_acceleration >= -0.1640738089005461
 CBF_Constraint_obs1: - 0.6717131914771827 x1_input_acceleration
   + 0.00553263107913449 x2_input_acceleration
   + 0.4128296180961009 x3_input_acceleration >= -0.7270718939856158
 CBF_Constraint_for_table: x3_input_acceleration >= -1.271191703378527
 Relaxed_CLF_constraint: - 0.1140074117786305 x1_input_acceleration
   + 0.1585084734756634 x2_input_acceleration
   + 0.2071616724132238 x3_input_acceleration + relaxation_CLF
   <= -20.03241543384843
 CBF_Constraint_obs2: - 0.3518638190937934 x1_input_acceleration
   + 0.00253263107919599 x2_input_acceleration
   + 0.4154742616745531 x3_input_acceleration >= -0.3205417794018351
 CBF_Constraint_obs3: 0.0281361809062066 x1_input_acceleration
   + 0.00253263107919599 x2_input_acceleration
   + 0.4154742616745531 x3_input_acceleration >= -0.1667711512627314
 CBF_Constraint_obs1: - 0.6718638190937934 x1_input_acceleration
   + 0.00253263107919599 x2_input_acceleration
   + 0.4154742616745531 x3_input_acceleration >= -0.7300328346768699
 CBF_Constraint_for_table: x3_input_acceleration >= -1.277803312324658
 Relaxed_CLF_constraint: - 0.1140827255869359 x1_input_acceleration
   + 0.1570084734756941 x2_input_acceleration
   + 0.2084839942024499 x3_input_acceleration + relaxation_CLF
   <= -20.03245284194365
 CBF_Constraint_obs2: - 0.3519753346480727 x1_input_acceleration
   - 0.000467369 x2_input_acceleration
   + 0.4181189052542525 x3_input_acceleration >= -0.3233878419566811
 CBF_Constraint_obs3: 0.0280246653519273 x1_input_acceleration
   - 0.000467369 x2_input_acceleration
   + 0.4181189052542525 x3_input_acceleration >= -0.1695112740410121
 CBF_Constraint_obs1: - 0.6719753346480728 x1_input_acceleration
   - 0.000467369 x2_input_acceleration
   + 0.4181189052542525 x3_input_acceleration >= -0.7329681096751395
 CBF_Constraint_for_table: x3_input_acceleration >= -1.284414921273906
 Relaxed_CLF_constraint: - 0.1141384833640755 x1_input_acceleration
   + 0.1555084734759835 x2_input_acceleration
   + 0.2098063159922996 x3_input_acceleration + relaxation_CLF
   <= -20.0324916675751
 CBF_Constraint_obs2: - 0.3520337590384353 x1_input_acceleration
   - 0.00346951468740408 x2_input_acceleration
   + 0.4207666482741141 x3_input_acceleration >= -0.3262304716684116
 CBF_Constraint_obs3: 0.0279662409615647 x1_input_acceleration
   - 0.00346951468740408 x2_input_acceleration
   + 0.4207666482741141 x3_input_acceleration >= -0.172298400581898
 CBF_Constraint_obs1: - 0.6720337590384353 x1_input_acceleration
   - 0.00346951468740408 x2_input_acceleration
   + 0.4207666482741141 x3_input_acceleration >= -0.7358574788991599
 CBF_Constraint_for_table: x3_input_acceleration >= -1.290683207465833
 Relaxed_CLF_constraint: - 0.1141746096980507 x1_input_acceleration
   + 0.1540084734760001 x2_input_acceleration
   + 0.2111286377814662 x3_input_acceleration + relaxation_CLF
   <= -20.03253190123743
 CBF_Constraint_obs2: - 0.3520866166203987 x1_input_acceleration
   - 0.00646295817634557 x2_input_acceleration
   + 0.4234357297571277 x3_input_acceleration >= -0.3291307158331751
 CBF_Constraint_obs3: 0.0279334105402459 x1_input_acceleration
   - 0.00646951468727086 x2_input_acceleration
   + 0.4234112918527698 x3_input_acceleration >= -0.1751240651414126
 CBF_Constraint_obs1: - 0.6720665894597542 x1_input_acceleration
   - 0.00646951468727086 x2_input_acceleration
   + 0.4234112918527698 x3_input_acceleration >= -0.7387405966959825
 CBF_Constraint_for_table: x3_input_acceleration >= -1.297710558809781
 Relaxed_CLF_constraint: - 0.1141910249087102 x1_input_acceleration
   + 0.1525084734760667 x2_input_acceleration
   + 0.2124509595707941 x3_input_acceleration + relaxation_CLF
   <= -20.0325735339497
 CBF_Constraint_obs2: - 0.3520798567311704 x1_input_acceleration
   - 0.00946295817612142 x2_input_acceleration
   + 0.4260803733361319 x3_input_acceleration >= -0.3319928220444275
 CBF_Constraint_obs3: 0.0279201432688296 x1_input_acceleration
   - 0.00946295817612142 x2_input_acceleration
   + 0.4260803733361319 x3_input_acceleration >= -0.1780169581498156
 CBF_Constraint_obs1: - 0.6720798567311703 x1_input_acceleration
   - 0.00946295817612142 x2_input_acceleration
   + 0.4260803733361319 x3_input_acceleration >= -0.7416567074293636
 CBF_Constraint_for_table: x3_input_acceleration >= -1.304322167757291
 Relaxed_CLF_constraint: - 0.114187644964096 x1_input_acceleration
   + 0.1510084734761787 x2_input_acceleration
   + 0.2137732813602962 x3_input_acceleration + relaxation_CLF
   <= -20.03261655725904
 CBF_Constraint_obs2: - 0.3520333295409654 x1_input_acceleration
   - 0.0124629581760456 x2_input_acceleration
   + 0.4287250169146441 x3_input_acceleration >= -0.3348599132033221
 CBF_Constraint_obs3: 0.0279666704590345 x1_input_acceleration
   - 0.0124629581760456 x2_input_acceleration
   + 0.4287250169146441 x3_input_acceleration >= -0.1809282501394049
 CBF_Constraint_obs1: - 0.6720333295409655 x1_input_acceleration
   - 0.0124629581760456 x2_input_acceleration
   + 0.4287250169146441 x3_input_acceleration >= -0.7444865768360944
 CBF_Constraint_for_table: x3_input_acceleration >= -1.310929604378438
 Relaxed_CLF_constraint: - 0.1141643813689936 x1_input_acceleration
   + 0.1495084734762167 x2_input_acceleration
   + 0.2150956031495523 x3_input_acceleration + relaxation_CLF
   <= -20.03266096324353
 CBF_Constraint_obs2: - 0.3519382658636594 x1_input_acceleration
   - 0.0154736870119941 x2_input_acceleration
   + 0.4313472491468052 x3_input_acceleration >= -0.3377005341442262
 CBF_Constraint_obs3: 0.0280617341363406 x1_input_acceleration
   - 0.0154736870119941 x2_input_acceleration
   + 0.4313472491468052 x3_input_acceleration >= -0.1838591815737498
 CBF_Constraint_obs1: - 0.671946848932507 x1_input_acceleration
   - 0.0154629581759346 x2_input_acceleration
   + 0.4313696604932407 x3_input_acceleration >= -0.7472893185749123
 CBF_Constraint_for_table: x3_input_acceleration >= -1.317541213324929
 Relaxed_CLF_constraint: - 0.1141211410647643 x1_input_acceleration
   + 0.1480084734762722 x2_input_acceleration
   + 0.2164179249388506 x3_input_acceleration + relaxation_CLF
   <= -20.03270674451742
 CBF_Constraint_obs2: - 0.351811636353077 x1_input_acceleration
   - 0.0184736870109393 x2_input_acceleration
   + 0.4339918927286754 x3_input_acceleration >= -0.3405770844242824
 CBF_Constraint_obs3: 0.028188363646923 x1_input_acceleration
   - 0.0184736870109393 x2_input_acceleration
   + 0.4339918927286754 x3_input_acceleration >= -0.1868560298888592
 CBF_Constraint_obs1: - 0.6718116363530771 x1_input_acceleration
   - 0.0184736870109393 x2_input_acceleration
   + 0.4339918927286754 x3_input_acceleration >= -0.750026393506744
 CBF_Constraint_for_table: x3_input_acceleration >= -1.324152822279605
 Relaxed_CLF_constraint: - 0.1140578263094731 x1_input_acceleration
   + 0.1465084734767996 x2_input_acceleration
   + 0.2177402467297857 x3_input_acceleration + relaxation_CLF
   <= -20.03275389423556
 CBF_Constraint_obs2: - 0.351713317400823 x1_input_acceleration
   - 0.0218438318548989 x2_input_acceleration
   + 0.4364177872609065 x3_input_acceleration >= -0.3432998695811286
 CBF_Constraint_obs3: 0.028286682599177 x1_input_acceleration
   - 0.0218438318548989 x2_input_acceleration
   + 0.4364177872609065 x3_input_acceleration >= -0.1896722180503469
 CBF_Constraint_obs1: - 0.671713317400823 x1_input_acceleration
   - 0.0218438318548989 x2_input_acceleration
   + 0.4364177872609065 x3_input_acceleration >= -0.7526705235017871
 CBF_Constraint_for_table: x3_input_acceleration >= -1.330363291966658
 Relaxed_CLF_constraint: - 0.1139743345579555 x1_input_acceleration
   + 0.1450084734768474 x2_input_acceleration
   + 0.2190625685190641 x3_input_acceleration + relaxation_CLF
   <= -20.03280240609767
 CBF_Constraint_obs2: - 0.3515653696072736 x1_input_acceleration
   - 0.0249238212881311 x2_input_acceleration
   + 0.4389770769879571 x3_input_acceleration >= -0.3461503501194906
 CBF_Constraint_obs3: 0.0284346303927264 x1_input_acceleration
   - 0.0249238212881311 x2_input_acceleration
   + 0.4389770769879571 x3_input_acceleration >= -0.1926632489925806
 CBF_Constraint_obs1: - 0.6715653696072735 x1_input_acceleration
   - 0.0249238212881311 x2_input_acceleration
   + 0.4389770769879571 x3_input_acceleration >= -0.7554026458053092
 CBF_Constraint_for_table: x3_input_acceleration >= -1.33737305993968
 Relaxed_CLF_constraint: - 0.1138705583387931 x1_input_acceleration
   + 0.1435084734768756 x2_input_acceleration
   + 0.2203848903082486 x3_input_acceleration + relaxation_CLF
   <= -20.03285227435689
 CBF_Constraint_obs2: - 0.3513170230665617 x1_input_acceleration
   - 0.0279238212814288 x2_input_acceleration
   + 0.4416217205836605 x3_input_acceleration >= -0.3490414182282262
 CBF_Constraint_obs3: 0.0286829769334383 x1_input_acceleration
   - 0.0279238212814288 x2_input_acceleration
   + 0.4416217205836605 x3_input_acceleration >= -0.1957902463149926
 CBF_Constraint_obs1: - 0.6713170230665617 x1_input_acceleration
   - 0.0279238212814288 x2_input_acceleration
   + 0.4416217205836605 x3_input_acceleration >= -0.7580950366814755
 CBF_Constraint_for_table: x3_input_acceleration >= -1.343984668928939
 Relaxed_CLF_constraint: - 0.1137463850684372 x1_input_acceleration
   + 0.1420084734802267 x2_input_acceleration
   + 0.2217072121061003 x3_input_acceleration + relaxation_CLF
   <= -20.03290349382247
 CBF_Constraint_obs2: - 0.3510276466786042 x1_input_acceleration
   - 0.0309238212622019 x2_input_acceleration
   + 0.444266364220043 x3_input_acceleration >= -0.3519366172893394
 CBF_Constraint_obs3: 0.0289723533213958 x1_input_acceleration
   - 0.0309238212622019 x2_input_acceleration
   + 0.444266364220043 x3_input_acceleration >= -0.1989603529446654
 CBF_Constraint_obs1: - 0.6710276466786043 x1_input_acceleration
   - 0.0309238212622019 x2_input_acceleration
   + 0.444266364220043 x3_input_acceleration >= -0.7607587346322229
 CBF_Constraint_for_table: x3_input_acceleration >= -1.350135236092558
 Relaxed_CLF_constraint: - 0.1136016968744584 x1_input_acceleration
   + 0.1405084734898401 x2_input_acceleration
   + 0.2230295339242915 x3_input_acceleration + relaxation_CLF
   <= -20.03295605986307
 CBF_Constraint_obs2: - 0.3506540795063957 x1_input_acceleration
   - 0.0338261888537107 x2_input_acceleration
   + 0.4469207829608828 x3_input_acceleration >= -0.3548008509615164
 CBF_Constraint_obs3: 0.0293459204936044 x1_input_acceleration
   - 0.0338261888537107 x2_input_acceleration
   + 0.4469207829608828 x3_input_acceleration >= -0.2021794754304405
 CBF_Constraint_obs1: - 0.6706540795063957 x1_input_acceleration
   - 0.0338261888537107 x2_input_acceleration
   + 0.4469207829608828 x3_input_acceleration >= -0.7633241145666327
 CBF_Constraint_for_table: x3_input_acceleration >= -1.356746845040299
 Relaxed_CLF_constraint: - 0.1134363709604733 x1_input_acceleration
   + 0.1390084734900147 x2_input_acceleration
   + 0.2243518557138399 x3_input_acceleration + relaxation_CLF
   <= -20.03300996844838
 CBF_Constraint_obs2: - 0.3502818946061825 x1_input_acceleration
   - 0.0368261888453643 x2_input_acceleration
   + 0.4495654265601657 x3_input_acceleration >= -0.3577033082899936
 CBF_Constraint_obs3: 0.0297181053938175 x1_input_acceleration
   - 0.0368261888453643 x2_input_acceleration
   + 0.4495654265601657 x3_input_acceleration >= -0.2054355084141202
 CBF_Constraint_obs1: - 0.6702818946061825 x1_input_acceleration
   - 0.0368261888453643 x2_input_acceleration
   + 0.4495654265601657 x3_input_acceleration >= -0.7659288239749396
 CBF_Constraint_for_table: x3_input_acceleration >= -1.363358454038506
 Relaxed_CLF_constraint: - 0.1132502785103667 x1_input_acceleration
   + 0.1375084734941879 x2_input_acceleration
   + 0.2256741775134813 x3_input_acceleration + relaxation_CLF
   <= -20.03306521610471
 CBF_Constraint_obs2: - 0.3497925677064379 x1_input_acceleration
   - 0.039472494883136 x2_input_acceleration
   + 0.4524990334565884 x3_input_acceleration >= -0.3608353669426361
 CBF_Constraint_obs3: 0.0302074322935621 x1_input_acceleration
   - 0.039472494883136 x2_input_acceleration
   + 0.4524990334565884 x3_input_acceleration >= -0.2090324276215201
 CBF_Constraint_obs1: - 0.6697925677064379 x1_input_acceleration
   - 0.039472494883136 x2_input_acceleration
   + 0.4524990334565884 x3_input_acceleration >= -0.7686694211077865
 CBF_Constraint_for_table: x3_input_acceleration >= -1.370045462860526
 Relaxed_CLF_constraint: - 0.1130432851959925 x1_input_acceleration
   + 0.1360084734942535 x2_input_acceleration
   + 0.2269964993027571 x3_input_acceleration + relaxation_CLF
   <= -20.03312179995434
 CBF_Constraint_obs2: - 0.3494867025147643 x1_input_acceleration
   - 0.0428701770648765 x2_input_acceleration
   + 0.454810606292034 x3_input_acceleration >= -0.363539368639931
 CBF_Constraint_obs3: 0.0305132974852357 x1_input_acceleration
   - 0.0428701770648765 x2_input_acceleration
   + 0.454810606292034 x3_input_acceleration >= -0.2120270012509049
 CBF_Constraint_obs1: - 0.6694867025147643 x1_input_acceleration
   - 0.0428701770648765 x2_input_acceleration
   + 0.454810606292034 x3_input_acceleration >= -0.7711287306517424
 CBF_Constraint_for_table: x3_input_acceleration >= -1.376636508234205
 Relaxed_CLF_constraint: - 0.1128152507477387 x1_input_acceleration
   + 0.1345084734983539 x2_input_acceleration
   + 0.2283188210979823 x3_input_acceleration + relaxation_CLF
   <= -20.03317971772468
 CBF_Constraint_obs2: - 0.3489882590500885 x1_input_acceleration
   - 0.0458701770648238 x2_input_acceleration
   + 0.4574552498704217 x3_input_acceleration >= -0.3664527296659749
 CBF_Constraint_obs3: 0.0310117409499115 x1_input_acceleration
   - 0.0458701770648238 x2_input_acceleration
   + 0.4574552498704217 x3_input_acceleration >= -0.2154138835683908
 CBF_Constraint_obs1: - 0.6689882590500884 x1_input_acceleration
   - 0.0458701770648238 x2_input_acceleration
   + 0.4574552498704217 x3_input_acceleration >= -0.7736433369060456
 CBF_Constraint_for_table: x3_input_acceleration >= -1.383248117180174
 Relaxed_CLF_constraint: - 0.1125660290154007 x1_input_acceleration
   + 0.1330084734983803 x2_input_acceleration
   + 0.2296411428871762 x3_input_acceleration + relaxation_CLF
   <= -20.03323896776688
 CBF_Constraint_obs2: - 0.3483760869884991 x1_input_acceleration
   - 0.048502774032935 x2_input_acceleration
   + 0.4603856381200309 x3_input_acceleration >= -0.3695916910768704
 CBF_Constraint_obs3: 0.0316239130115009 x1_input_acceleration
   - 0.048502774032935 x2_input_acceleration
   + 0.4603856381200309 x3_input_acceleration >= -0.2191344084377962
 CBF_Constraint_obs1: - 0.6684471357250714 x1_input_acceleration
   - 0.0488701770633305 x2_input_acceleration
   + 0.4600998934529777 x3_input_acceleration >= -0.7761272242758683
 CBF_Constraint_for_table: x3_input_acceleration >= -1.389895190900206
 Relaxed_CLF_constraint: - 0.1122954673528922 x1_input_acceleration
   + 0.1315084734991269 x2_input_acceleration
   + 0.2309634646784542 x3_input_acceleration + relaxation_CLF
   <= -20.03329954904254
 CBF_Constraint_obs2: - 0.3477919660817288 x1_input_acceleration
   - 0.0515027740264132 x2_input_acceleration
   + 0.4630302817051803 x3_input_acceleration >= -0.3725110364742359
 CBF_Constraint_obs3: 0.0322080339182712 x1_input_acceleration
   - 0.0515027740264132 x2_input_acceleration
   + 0.4630302817051803 x3_input_acceleration >= -0.2226086686965936
 CBF_Constraint_obs1: - 0.6677919660817289 x1_input_acceleration
   - 0.0515027740264132 x2_input_acceleration
   + 0.4630302817051803 x3_input_acceleration >= -0.7787446093396191
 CBF_Constraint_for_table: x3_input_acceleration >= -1.396506799863079
 Relaxed_CLF_constraint: - 0.1120034068995071 x1_input_acceleration
   + 0.1300084735023878 x2_input_acceleration
   + 0.2322857864710289 x3_input_acceleration + relaxation_CLF
   <= -20.03336146117439
 CBF_Constraint_obs2: - 0.3472241200156178 x1_input_acceleration
   - 0.0548744685677767 x2_input_acceleration
   + 0.4654055123350773 x3_input_acceleration >= -0.3752226096663647
 CBF_Constraint_obs3: 0.0327758799843822 x1_input_acceleration
   - 0.0548744685677767 x2_input_acceleration
   + 0.4654055123350773 x3_input_acceleration >= -0.2258596956515278
 CBF_Constraint_obs1: - 0.6672241200156178 x1_input_acceleration
   - 0.0548744685677767 x2_input_acceleration
   + 0.4654055123350773 x3_input_acceleration >= -0.7810019056788589
 CBF_Constraint_for_table: x3_input_acceleration >= -1.403042711024919
 Relaxed_CLF_constraint: - 0.1116896815440638 x1_input_acceleration
   + 0.1285084735141158 x2_input_acceleration
   + 0.2336081082831697 x3_input_acceleration + relaxation_CLF
   <= -20.03342470439359
 CBF_Constraint_obs2: - 0.346564439614964 x1_input_acceleration
   - 0.0578763759163348 x2_input_acceleration
   + 0.4680153468008675 x3_input_acceleration >= -0.378118688169974
 CBF_Constraint_obs3: 0.033435560385036 x1_input_acceleration
   - 0.0578763759163348 x2_input_acceleration
   + 0.4680153468008675 x3_input_acceleration >= -0.2293824705357582
 CBF_Constraint_obs1: - 0.666564439614964 x1_input_acceleration
   - 0.0578763759163348 x2_input_acceleration
   + 0.4680153468008675 x3_input_acceleration >= -0.7833702398619452
 CBF_Constraint_for_table: x3_input_acceleration >= -1.409682036130587
 Relaxed_CLF_constraint: - 0.1113541192978385 x1_input_acceleration
   + 0.1270084735141531 x2_input_acceleration
   + 0.2349304300723393 x3_input_acceleration + relaxation_CLF
   <= -20.03348927968119
 CBF_Constraint_obs2: - 0.345849281029486 x1_input_acceleration
   - 0.060876375901414 x2_input_acceleration
   + 0.470659990407083 x3_input_acceleration >= -0.3810481061268727
 CBF_Constraint_obs3: 0.034150718970514 x1_input_acceleration
   - 0.060876375901414 x2_input_acceleration
   + 0.470659990407083 x3_input_acceleration >= -0.2329912891488609
 CBF_Constraint_obs1: - 0.665849281029486 x1_input_acceleration
   - 0.060876375901414 x2_input_acceleration
   + 0.470659990407083 x3_input_acceleration >= -0.7857275309504614
 CBF_Constraint_for_table: x3_input_acceleration >= -1.416293645146125
 Relaxed_CLF_constraint: - 0.1109965400050995 x1_input_acceleration
   + 0.1255084735216135 x2_input_acceleration
   + 0.236252751875447 x3_input_acceleration + relaxation_CLF
   <= -20.03355518863502
 CBF_Constraint_obs2: - 0.3450897139057526 x1_input_acceleration
   - 0.0638763758970016 x2_input_acceleration
   + 0.4733046339957546 x3_input_acceleration >= -0.3839804732539305
 CBF_Constraint_obs3: 0.0349102860942474 x1_input_acceleration
   - 0.0638763758970016 x2_input_acceleration
   + 0.4733046339957546 x3_input_acceleration >= -0.2366452450434655
 CBF_Constraint_obs1: - 0.6650897139057526 x1_input_acceleration
   - 0.0638763758970016 x2_input_acceleration
   + 0.4733046339957546 x3_input_acceleration >= -0.7880522443785325
 CBF_Constraint_for_table: x3_input_acceleration >= -1.422905254117804
 Relaxed_CLF_constraint: - 0.1106167564432328 x1_input_acceleration
   + 0.1240084735238197 x2_input_acceleration
   + 0.2375750736697828 x3_input_acceleration + relaxation_CLF
   <= -20.03362243357637
 CBF_Constraint_obs2: - 0.3441961797151056 x1_input_acceleration
   - 0.0668409707377706 x2_input_acceleration
   + 0.47599564998808 x3_input_acceleration >= -0.3868882303840194
 CBF_Constraint_obs3: 0.0358038202848944 x1_input_acceleration
   - 0.0668409707377706 x2_input_acceleration
   + 0.47599564998808 x3_input_acceleration >= -0.2404018596546691
 CBF_Constraint_obs1: - 0.6641961797151056 x1_input_acceleration
   - 0.0668409707377706 x2_input_acceleration
   + 0.47599564998808 x3_input_acceleration >= -0.790245174156104
 CBF_Constraint_for_table: x3_input_acceleration >= -1.429519545273545
 Relaxed_CLF_constraint: - 0.1102145736222013 x1_input_acceleration
   + 0.1225084735239369 x2_input_acceleration
   + 0.2388973954591279 x3_input_acceleration + relaxation_CLF
   <= -20.03369101755242
 CBF_Constraint_obs2: - 0.3433466094813176 x1_input_acceleration
   - 0.0698409707377269 x2_input_acceleration
   + 0.4786402935663787 x3_input_acceleration >= -0.389826482576517
 CBF_Constraint_obs3: 0.0366533905186824 x1_input_acceleration
   - 0.0698409707377269 x2_input_acceleration
   + 0.4786402935663787 x3_input_acceleration >= -0.2441472035692653
 CBF_Constraint_obs1: - 0.6633466094813176 x1_input_acceleration
   - 0.0698409707377269 x2_input_acceleration
   + 0.4786402935663787 x3_input_acceleration >= -0.7925037701615711
 CBF_Constraint_for_table: x3_input_acceleration >= -1.436131154219291
 Relaxed_CLF_constraint: - 0.1097897885053073 x1_input_acceleration
   + 0.1210084735239588 x2_input_acceleration
   + 0.2402197172482772 x3_input_acceleration + relaxation_CLF
   <= -20.03376094435179
 CBF_Constraint_obs2: - 0.342450458253869 x1_input_acceleration
   - 0.0728172480890251 x2_input_acceleration
   + 0.4813044874682923 x3_input_acceleration >= -0.3927858470433238
 CBF_Constraint_obs3: 0.037549541746131 x1_input_acceleration
   - 0.0728172480890251 x2_input_acceleration
   + 0.4813044874682923 x3_input_acceleration >= -0.2479579117021483
 CBF_Constraint_obs1: - 0.6624504582538691 x1_input_acceleration
   - 0.0728172480890251 x2_input_acceleration
   + 0.4813044874682923 x3_input_acceleration >= -0.7947462136464191
 CBF_Constraint_for_table: x3_input_acceleration >= -1.442716239098435
 Relaxed_CLF_constraint: - 0.1093421897287412 x1_input_acceleration
   + 0.1195084735239993 x2_input_acceleration
   + 0.2415420390374909 x3_input_acceleration + relaxation_CLF
   <= -20.03383221852844
 CBF_Constraint_obs2: - 0.3414505423395484 x1_input_acceleration
   - 0.0754771439858863 x2_input_acceleration
   + 0.4842090072978768 x3_input_acceleration >= -0.3959295435957916
 CBF_Constraint_obs3: 0.0385494576604516 x1_input_acceleration
   - 0.0754771439858863 x2_input_acceleration
   + 0.4842090072978768 x3_input_acceleration >= -0.2520515283732206
 CBF_Constraint_obs1: - 0.6614505423395484 x1_input_acceleration
   - 0.0754771439858863 x2_input_acceleration
   + 0.4842090072978768 x3_input_acceleration >= -0.7970899774674302
 CBF_Constraint_for_table: x3_input_acceleration >= -1.449349901762911
 Relaxed_CLF_constraint: - 0.1088715572568104 x1_input_acceleration
   + 0.1180084735240245 x2_input_acceleration
   + 0.2428643608266728 x3_input_acceleration + relaxation_CLF
   <= -20.03390484542509
 CBF_Constraint_obs2: - 0.3404627518606507 x1_input_acceleration
   - 0.0784771439843187 x2_input_acceleration
   + 0.4868536508800557 x3_input_acceleration >= -0.398875031134627
 CBF_Constraint_obs3: 0.0395372481393493 x1_input_acceleration
   - 0.0784771439843187 x2_input_acceleration
   + 0.4868536508800557 x3_input_acceleration >= -0.2559354168670089
 CBF_Constraint_obs1: - 0.6604627518606507 x1_input_acceleration
   - 0.0784771439843187 x2_input_acceleration
   + 0.4868536508800557 x3_input_acceleration >= -0.7992452326231475
 CBF_Constraint_for_table: x3_input_acceleration >= -1.455961510718358
 Relaxed_CLF_constraint: - 0.1083776620173615 x1_input_acceleration
   + 0.1165084735248084 x2_input_acceleration
   + 0.2441866826177622 x3_input_acceleration + relaxation_CLF
   <= -20.03397883119812
 CBF_Constraint_obs2: - 0.3394279585461444 x1_input_acceleration
   - 0.081477143977012 x2_input_acceleration
   + 0.4894982944754795 x3_input_acceleration >= -0.4018230554098213
 CBF_Constraint_obs3: 0.0405720414538556 x1_input_acceleration
   - 0.081477143977012 x2_input_acceleration
   + 0.4894982944754795 x3_input_acceleration >= -0.2598664947909842
 CBF_Constraint_obs1: - 0.6594279585461444 x1_input_acceleration
   - 0.081477143977012 x2_input_acceleration
   + 0.4894982944754795 x3_input_acceleration >= -0.801365422246737
 CBF_Constraint_for_table: x3_input_acceleration >= -1.462573119706918
 Relaxed_CLF_constraint: - 0.1078602653601084 x1_input_acceleration
   + 0.1150084735284617 x2_input_acceleration
   + 0.2455090044154741 x3_input_acceleration + relaxation_CLF
   <= -20.03405418283039
 CBF_Constraint_obs2: - 0.3381420571267712 x1_input_acceleration
   - 0.0847544247839721 x2_input_acceleration
   + 0.491968415654678 x3_input_acceleration >= -0.4044453566502007
 CBF_Constraint_obs3: 0.0418579428732288 x1_input_acceleration
   - 0.0847544247839721 x2_input_acceleration
   + 0.491968415654678 x3_input_acceleration >= -0.2637104023797681
 CBF_Constraint_obs1: - 0.6581420571267712 x1_input_acceleration
   - 0.0847544247839721 x2_input_acceleration
   + 0.491968415654678 x3_input_acceleration >= -0.8029590023516177
 CBF_Constraint_for_table: x3_input_acceleration >= -1.469173105748939
 Relaxed_CLF_constraint: - 0.1073191193836982 x1_input_acceleration
   + 0.1135084735287292 x2_input_acceleration
   + 0.2468313262050246 x3_input_acceleration + relaxation_CLF
   <= -20.0341309082177
 CBF_Constraint_obs2: - 0.3370117491618162 x1_input_acceleration
   - 0.0877544247837848 x2_input_acceleration
   + 0.4946130592334387 x3_input_acceleration >= -0.4073997956331263
 CBF_Constraint_obs3: 0.0429882508381838 x1_input_acceleration
   - 0.0877544247837848 x2_input_acceleration
   + 0.4946130592334387 x3_input_acceleration >= -0.2677386339294009
 CBF_Constraint_obs1: - 0.6570117491618161 x1_input_acceleration
   - 0.0877544247837848 x2_input_acceleration
   + 0.4946130592334387 x3_input_acceleration >= -0.8050091949625792
 CBF_Constraint_for_table: x3_input_acceleration >= -1.47578471469584
 Relaxed_CLF_constraint: - 0.1067539654012207 x1_input_acceleration
   + 0.1120084735288229 x2_input_acceleration
   + 0.2481536479944049 x3_input_acceleration + relaxation_CLF
   <= -20.03420901611363
 CBF_Constraint_obs2: - 0.3360245749673739 x1_input_acceleration
   - 0.0904590241543489 x2_input_acceleration
   + 0.4974526100235322 x3_input_acceleration >= -0.4106930615652322
 CBF_Constraint_obs3: 0.0439754250326261 x1_input_acceleration
   - 0.0904590241543489 x2_input_acceleration
   + 0.4974526100235322 x3_input_acceleration >= -0.271969715346227
 CBF_Constraint_obs1: - 0.656024574967374 x1_input_acceleration
   - 0.0904590241543489 x2_input_acceleration
   + 0.4974526100235322 x3_input_acceleration >= -0.8075127215391315
 CBF_Constraint_for_table: x3_input_acceleration >= -1.482410330791557
 Relaxed_CLF_constraint: - 0.1061645340352007 x1_input_acceleration
   + 0.1105084735337874 x2_input_acceleration
   + 0.2494759697952439 x3_input_acceleration + relaxation_CLF
   <= -20.03428851620598
 CBF_Constraint_obs2: - 0.3349549075647771 x1_input_acceleration
   - 0.0938863894557366 x2_input_acceleration
   + 0.4997615602435411 x3_input_acceleration >= -0.4134638266547764
 CBF_Constraint_obs3: 0.0450450924352229 x1_input_acceleration
   - 0.0938863894557366 x2_input_acceleration
   + 0.4997615602435411 x3_input_acceleration >= -0.2757566644682382
 CBF_Constraint_obs1: - 0.6549549075647771 x1_input_acceleration
   - 0.0938863894557366 x2_input_acceleration
   + 0.4997615602435411 x3_input_acceleration >= -0.8094277527065981
 CBF_Constraint_for_table: x3_input_acceleration >= -1.489383143887356
 Relaxed_CLF_constraint: - 0.1055505453656406 x1_input_acceleration
   + 0.1090084735346133 x2_input_acceleration
   + 0.2507982915849358 x3_input_acceleration + relaxation_CLF
   <= -20.0343694191965
 CBF_Constraint_obs2: - 0.3336772312107266 x1_input_acceleration
   - 0.0968863894553513 x2_input_acceleration
   + 0.5024062038225907 x3_input_acceleration >= -0.4164243259119715
 CBF_Constraint_obs3: 0.0463227687892734 x1_input_acceleration
   - 0.0968863894553513 x2_input_acceleration
   + 0.5024062038225907 x3_input_acceleration >= -0.2799309562617813
 CBF_Constraint_obs1: - 0.6536772312107266 x1_input_acceleration
   - 0.0968863894553513 x2_input_acceleration
   + 0.5024062038225907 x3_input_acceleration >= -0.8113661108805528
 CBF_Constraint_for_table: x3_input_acceleration >= -1.49599475283498
 Relaxed_CLF_constraint: - 0.1049117071886153 x1_input_acceleration
   + 0.107508473534806 x2_input_acceleration
   + 0.2521206133744607 x3_input_acceleration + relaxation_CLF
   <= -20.03445173675013
 CBF_Constraint_obs2: - 0.3322848743938673 x1_input_acceleration
   - 0.0998443085752717 x2_input_acceleration
   + 0.5050751660971664 x3_input_acceleration >= -0.4193538088923781
 CBF_Constraint_obs3: 0.0477151256061327 x1_input_acceleration
   - 0.0998443085752717 x2_input_acceleration
   + 0.5050751660971664 x3_input_acceleration >= -0.2841831782182041
 CBF_Constraint_obs1: - 0.6522848743938674 x1_input_acceleration
   - 0.0998443085752717 x2_input_acceleration
   + 0.5050751660971664 x3_input_acceleration >= -0.8131817084074721
 CBF_Constraint_for_table: x3_input_acceleration >= -1.502198069967037
 Relaxed_CLF_constraint: - 0.1042477152883644 x1_input_acceleration
   + 0.1060084735352401 x2_input_acceleration
   + 0.2534429351642143 x3_input_acceleration + relaxation_CLF
   <= -20.03453548159551
 CBF_Constraint_obs2: - 0.3309059492547446 x1_input_acceleration
   - 0.1028443085752337 x2_input_acceleration
   + 0.5077198096754789 x3_input_acceleration >= -0.4223188802442576
 CBF_Constraint_obs3: 0.0490940507452554 x1_input_acceleration
   - 0.1028443085752337 x2_input_acceleration
   + 0.5077198096754789 x3_input_acceleration >= -0.2884582284522502
 CBF_Constraint_obs1: - 0.6509059492547447 x1_input_acceleration
   - 0.1028443085752337 x2_input_acceleration
   + 0.5077198096754789 x3_input_acceleration >= -0.8150436396480533
 CBF_Constraint_for_table: x3_input_acceleration >= -1.508809678912818
 Relaxed_CLF_constraint: - 0.103558252718803 x1_input_acceleration
   + 0.104508473535259 x2_input_acceleration
   + 0.2547652569533705 x3_input_acceleration + relaxation_CLF
   <= -20.03462066755894
 CBF_Constraint_obs2: - 0.3294568252798576 x1_input_acceleration
   - 0.1058199898796633 x2_input_acceleration
   + 0.5103966397628878 x3_input_acceleration >= -0.4253054998285772
 CBF_Constraint_obs3: 0.0505431747201424 x1_input_acceleration
   - 0.1058199898796633 x2_input_acceleration
   + 0.5103966397628878 x3_input_acceleration >= -0.2928215158127125
 CBF_Constraint_obs1: - 0.6494754219290275 x1_input_acceleration
   - 0.1058443085747317 x2_input_acceleration
   + 0.5103644532547091 x3_input_acceleration >= -0.8168665206169118
 CBF_Constraint_for_table: x3_input_acceleration >= -1.51589603885653
 Relaxed_CLF_constraint: - 0.1028429890559445 x1_input_acceleration
   + 0.1030084735355101 x2_input_acceleration
   + 0.2560875787429857 x3_input_acceleration + relaxation_CLF
   <= -20.03470730960181
 CBF_Constraint_obs2: - 0.3279740068486701 x1_input_acceleration
   - 0.1088199898534976 x2_input_acceleration
   + 0.5130412834011773 x3_input_acceleration >= -0.4282751222925173
 CBF_Constraint_obs3: 0.05202599315133 x1_input_acceleration
   - 0.1088199898534976 x2_input_acceleration
   + 0.5130412834011773 x3_input_acceleration >= -0.2971998157862808
 CBF_Constraint_obs1: - 0.64797400684867 x1_input_acceleration
   - 0.1088199898534976 x2_input_acceleration
   + 0.5130412834011773 x3_input_acceleration >= -0.8186543277714533
 CBF_Constraint_for_table: x3_input_acceleration >= -1.522035877188856
 Relaxed_CLF_constraint: - 0.1021015798403507 x1_input_acceleration
   + 0.1015084735485929 x2_input_acceleration
   + 0.2574099005621304 x3_input_acceleration + relaxation_CLF
   <= -20.03479542388619
 CBF_Constraint_obs2: - 0.3265201950253229 x1_input_acceleration
   - 0.1118677927783704 x2_input_acceleration
   + 0.515624653405014 x3_input_acceleration >= -0.4312482800244001
 CBF_Constraint_obs3: 0.0534798049746771 x1_input_acceleration
   - 0.1118677927783704 x2_input_acceleration
   + 0.515624653405014 x3_input_acceleration >= -0.3015540947503433
 CBF_Constraint_obs1: - 0.6465201950253229 x1_input_acceleration
   - 0.1118677927783704 x2_input_acceleration
   + 0.515624653405014 x3_input_acceleration >= -0.8204644360446584
 CBF_Constraint_for_table: x3_input_acceleration >= -1.528647486135521
 Relaxed_CLF_constraint: - 0.1013336659330717 x1_input_acceleration
   + 0.1000084735487115 x2_input_acceleration
   + 0.2587322223514633 x3_input_acceleration + relaxation_CLF
   <= -20.03488502780636
 CBF_Constraint_obs2: - 0.3249306082804935 x1_input_acceleration
   - 0.114867792777959 x2_input_acceleration
   + 0.5182692969840605 x3_input_acceleration >= -0.4342219677644549
 CBF_Constraint_obs3: 0.0550693917195065 x1_input_acceleration
   - 0.114867792777959 x2_input_acceleration
   + 0.5182692969840605 x3_input_acceleration >= -0.3060378898979861
 CBF_Constraint_obs1: - 0.6449306082804935 x1_input_acceleration
   - 0.114867792777959 x2_input_acceleration
   + 0.5182692969840605 x3_input_acceleration >= -0.8221664543888497
 CBF_Constraint_for_table: x3_input_acceleration >= -1.535259095083137
 Relaxed_CLF_constraint: - 0.100538872560657 x1_input_acceleration
   + 0.0985084735489172 x2_input_acceleration
   + 0.2600545441409866 x3_input_acceleration + relaxation_CLF
   <= -20.03497614007403
 CBF_Constraint_obs2: - 0.3232922025520284 x1_input_acceleration
   - 0.1178601633830688 x2_input_acceleration
   + 0.520915251865119 x3_input_acceleration >= -0.4372019574616571
 CBF_Constraint_obs3: 0.0567077974479716 x1_input_acceleration
   - 0.1178601633830688 x2_input_acceleration
   + 0.520915251865119 x3_input_acceleration >= -0.3105743650372302
 CBF_Constraint_obs1: - 0.6432922025520285 x1_input_acceleration
   - 0.1178601633830688 x2_input_acceleration
   + 0.520915251865119 x3_input_acceleration >= -0.8238357195032799
 CBF_Constraint_for_table: x3_input_acceleration >= -1.541904082631395
 Relaxed_CLF_constraint: - 0.0997168086734752 x1_input_acceleration
   + 0.0970084735490966 x2_input_acceleration
   + 0.2613768659304233 x3_input_acceleration + relaxation_CLF
   <= -20.03506878076638
 CBF_Constraint_obs2: - 0.321592717289519 x1_input_acceleration
   - 0.1208601633829959 x2_input_acceleration
   + 0.5235598954434926 x3_input_acceleration >= -0.4401800237792772
 CBF_Constraint_obs3: 0.058407282710481 x1_input_acceleration
   - 0.1208601633829959 x2_input_acceleration
   + 0.5235598954434926 x3_input_acceleration >= -0.3151669423542341
 CBF_Constraint_obs1: - 0.6415927172895191 x1_input_acceleration
   - 0.1208601633829959 x2_input_acceleration
   + 0.5235598954434926 x3_input_acceleration >= -0.8254541976108926
 CBF_Constraint_for_table: x3_input_acceleration >= -1.548515691577329
 Relaxed_CLF_constraint: - 0.0988670660422205 x1_input_acceleration
   + 0.0955084735491331 x2_input_acceleration
   + 0.26269918771961 x3_input_acceleration + relaxation_CLF
   <= -20.03516297139841
 CBF_Constraint_obs2: - 0.3197459446207005 x1_input_acceleration
   - 0.1235150524587477 x2_input_acceleration
   + 0.5265157945434018 x3_input_acceleration >= -0.4433903989861762
 CBF_Constraint_obs3: 0.0602540553792995 x1_input_acceleration
   - 0.1235150524587477 x2_input_acceleration
   + 0.5265157945434018 x3_input_acceleration >= -0.3201317515965107
 CBF_Constraint_obs1: - 0.6397459446207004 x1_input_acceleration
   - 0.1235150524587477 x2_input_acceleration
   + 0.5265157945434018 x3_input_acceleration >= -0.8271871546827364
 CBF_Constraint_for_table: x3_input_acceleration >= -1.555512048671584
 Relaxed_CLF_constraint: - 0.0979892176564196 x1_input_acceleration
   + 0.0940084735646324 x2_input_acceleration
   + 0.2640215095420561 x3_input_acceleration + relaxation_CLF
   <= -20.0352587349519
 CBF_Constraint_obs2: - 0.3179369627776751 x1_input_acceleration
   - 0.126538894316251 x2_input_acceleration
   + 0.5291327815669837 x3_input_acceleration >= -0.4463468807548568
 CBF_Constraint_obs3: 0.0620668519195905 x1_input_acceleration
   - 0.1265150524583408 x2_input_acceleration
   + 0.5291604381221595 x3_input_acceleration >= -0.3248364023492262
 CBF_Constraint_obs1: - 0.6379331480804095 x1_input_acceleration
   - 0.1265150524583408 x2_input_acceleration
   + 0.5291604381221595 x3_input_acceleration >= -0.8287194114899427
 CBF_Constraint_for_table: x3_input_acceleration >= -1.561748148356394
 Relaxed_CLF_constraint: - 0.0970828193862741 x1_input_acceleration
   + 0.0925084735648359 x2_input_acceleration
   + 0.265343831331435 x3_input_acceleration + relaxation_CLF
   <= -20.03535609613077
 CBF_Constraint_obs2: - 0.316066133858714 x1_input_acceleration
   - 0.1295388943119452 x2_input_acceleration
   + 0.5317774251540588 x3_input_acceleration >= -0.4493316950192955
 CBF_Constraint_obs3: 0.063933866141286 x1_input_acceleration
   - 0.1295388943119452 x2_input_acceleration
   + 0.5317774251540588 x3_input_acceleration >= -0.3295688678535172
 CBF_Constraint_obs1: - 0.636066133858714 x1_input_acceleration
   - 0.1295388943119452 x2_input_acceleration
   + 0.5317774251540588 x3_input_acceleration >= -0.8301846021062667
 CBF_Constraint_for_table: x3_input_acceleration >= -1.568359757324081
 Relaxed_CLF_constraint: - 0.0961474049267936 x1_input_acceleration
   + 0.0910084735669888 x2_input_acceleration
   + 0.2666661531249725 x3_input_acceleration + relaxation_CLF
   <= -20.03545508118305
 CBF_Constraint_obs2: - 0.3140766937508079 x1_input_acceleration
   - 0.1324896608634045 x2_input_acceleration
   + 0.5344608117789642 x3_input_acceleration >= -0.4523075489007101
 CBF_Constraint_obs3: 0.065923306249192 x1_input_acceleration
   - 0.1324896608634045 x2_input_acceleration
   + 0.5344608117789642 x3_input_acceleration >= -0.3344346898374426
 CBF_Constraint_obs1: - 0.6341362983955834 x1_input_acceleration
   - 0.132538894299989 x2_input_acceleration
   + 0.5344220687598602 x3_input_acceleration >= -0.8316279387783559
 CBF_Constraint_for_table: x3_input_acceleration >= -1.575370121412132
 Relaxed_CLF_constraint: - 0.0951824871952283 x1_input_acceleration
   + 0.0895084735729669 x2_input_acceleration
   + 0.2679884749278732 x3_input_acceleration + relaxation_CLF
   <= -20.03555571816168
 CBF_Constraint_obs2: - 0.3120868334873931 x1_input_acceleration
   - 0.1354896608632012 x2_input_acceleration
   + 0.5371054553575845 x3_input_acceleration >= -0.4552973875148641
 CBF_Constraint_obs3: 0.0679131665126069 x1_input_acceleration
   - 0.1354896608632012 x2_input_acceleration
   + 0.5371054553575845 x3_input_acceleration >= -0.3393148957018406
 CBF_Constraint_obs1: - 0.6320868334873931 x1_input_acceleration
   - 0.1354896608632012 x2_input_acceleration
   + 0.5371054553575845 x3_input_acceleration >= -0.8329668543047786
 CBF_Constraint_for_table: x3_input_acceleration >= -1.581981730358683
 Relaxed_CLF_constraint: - 0.0941875570635209 x1_input_acceleration
   + 0.0880084735730685 x2_input_acceleration
   + 0.2693107967171833 x3_input_acceleration + relaxation_CLF
   <= -20.03565803702188
 CBF_Constraint_obs2: - 0.3100358812044243 x1_input_acceleration
   - 0.1384896608628362 x2_input_acceleration
   + 0.5397500989365267 x3_input_acceleration >= -0.4582897538776221
 CBF_Constraint_obs3: 0.0699641187955757 x1_input_acceleration
   - 0.1384896608628362 x2_input_acceleration
   + 0.5397500989365267 x3_input_acceleration >= -0.344255666733419
 CBF_Constraint_obs1: - 0.6300358812044242 x1_input_acceleration
   - 0.1384896608628362 x2_input_acceleration
   + 0.5397500989365267 x3_input_acceleration >= -0.8343184588411613
 CBF_Constraint_for_table: x3_input_acceleration >= -1.588593339306038
 Relaxed_CLF_constraint: - 0.0931620809220364 x1_input_acceleration
   + 0.086508473573251 x2_input_acceleration
   + 0.2706331185066544 x3_input_acceleration + relaxation_CLF
   <= -20.03576206966173
 CBF_Constraint_obs2: - 0.308066724255982 x1_input_acceleration
   - 0.1418651701238918 x2_input_acceleration
   + 0.5421030373850677 x3_input_acceleration >= -0.4611331702877602
 CBF_Constraint_obs3: 0.0719332757440179 x1_input_acceleration
   - 0.1418651701238918 x2_input_acceleration
   + 0.5421030373850677 x3_input_acceleration >= -0.3489697822445773
 CBF_Constraint_obs1: - 0.6280667242559821 x1_input_acceleration
   - 0.1418651701238918 x2_input_acceleration
   + 0.5421030373850677 x3_input_acceleration >= -0.8355865496925458
 CBF_Constraint_for_table: x3_input_acceleration >= -1.595221339533531
 Relaxed_CLF_constraint: - 0.0921055 x1_input_acceleration
   + 0.0850084735737657 x2_input_acceleration
   + 0.2719554402966903 x3_input_acceleration + relaxation_CLF
   <= -20.03586785008934
 CBF_Constraint_obs2: - 0.3057414085977613 x1_input_acceleration
   - 0.1444732099795096 x2_input_acceleration
   + 0.5450681155342227 x3_input_acceleration >= -0.4643119598814444
 CBF_Constraint_obs3: 0.0742585914022387 x1_input_acceleration
   - 0.1444732099795096 x2_input_acceleration
   + 0.5450681155342227 x3_input_acceleration >= -0.3543576217135712
 CBF_Constraint_obs1: - 0.6257414085977613 x1_input_acceleration
   - 0.1444732099795096 x2_input_acceleration
   + 0.5450681155342227 x3_input_acceleration >= -0.8369050867596534
 CBF_Constraint_for_table: x3_input_acceleration >= -1.601832948480638
 Relaxed_CLF_constraint: - 0.0910172288044959 x1_input_acceleration
   + 0.0835084735739353 x2_input_acceleration
   + 0.2732777620861117 x3_input_acceleration + relaxation_CLF
   <= -20.03597541453948
 CBF_Constraint_obs2: - 0.3035779814381673 x1_input_acceleration
   - 0.1478127180357827 x2_input_acceleration
   + 0.5474130669591026 x3_input_acceleration >= -0.4670865703809612
 CBF_Constraint_obs3: 0.0764220185618328 x1_input_acceleration
   - 0.1478127180357827 x2_input_acceleration
   + 0.5474130669591026 x3_input_acceleration >= -0.3591874880147023
 CBF_Constraint_obs1: - 0.6235779814381672 x1_input_acceleration
   - 0.1478127180357827 x2_input_acceleration
   + 0.5474130669591026 x3_input_acceleration >= -0.837948955531495
 CBF_Constraint_for_table: x3_input_acceleration >= -1.607948944806357
 Relaxed_CLF_constraint: - 0.0898966529963054 x1_input_acceleration
   + 0.0820084735741191 x2_input_acceleration
   + 0.2746000838755169 x3_input_acceleration + relaxation_CLF
   <= -20.03608480160893
 CBF_Constraint_obs2: - 0.3012709314579717 x1_input_acceleration
   - 0.1508127180356391 x2_input_acceleration
   + 0.5500577105375364 x3_input_acceleration >= -0.470090168730809
 CBF_Constraint_obs3: 0.0787290685420283 x1_input_acceleration
   - 0.1508127180356391 x2_input_acceleration
   + 0.5500577105375364 x3_input_acceleration >= -0.364382783845736
 CBF_Constraint_obs1: - 0.6212709314579717 x1_input_acceleration
   - 0.1508127180356391 x2_input_acceleration
   + 0.5500577105375364 x3_input_acceleration >= -0.8391069138971865
 CBF_Constraint_for_table: x3_input_acceleration >= -1.614560553752441
 Relaxed_CLF_constraint: - 0.0887431280062076 x1_input_acceleration
   + 0.0805084735741909 x2_input_acceleration
   + 0.2759224056647337 x3_input_acceleration + relaxation_CLF
   <= -20.03619605241336
 CBF_Constraint_obs2: - 0.2988966291264428 x1_input_acceleration
   - 0.1538127180143283 x2_input_acceleration
   + 0.5527023541411396 x3_input_acceleration >= -0.4730967992490789
 CBF_Constraint_obs3: 0.0811033708735572 x1_input_acceleration
   - 0.1538127180143283 x2_input_acceleration
   + 0.5527023541411396 x3_input_acceleration >= -0.3696450015789583
 CBF_Constraint_obs1: - 0.6188966291264428 x1_input_acceleration
   - 0.1538127180143283 x2_input_acceleration
   + 0.5527023541411396 x3_input_acceleration >= -0.8402141025502332
 CBF_Constraint_for_table: x3_input_acceleration >= -1.621305081119298
 Relaxed_CLF_constraint: - 0.0875559768404432 x1_input_acceleration
   + 0.0790084735848463 x2_input_acceleration
   + 0.2772447274665353 x3_input_acceleration + relaxation_CLF
   <= -20.03630921075467
 CBF_Constraint_obs2: - 0.2964307629804535 x1_input_acceleration
   - 0.1565259004634976 x2_input_acceleration
   + 0.5556056818779933 x3_input_acceleration >= -0.4763365356152405
 CBF_Constraint_obs3: 0.0835692370195465 x1_input_acceleration
   - 0.1565259004634976 x2_input_acceleration
   + 0.5556056818779933 x3_input_acceleration >= -0.3752273107838097
 CBF_Constraint_obs1: - 0.6164307629804535 x1_input_acceleration
   - 0.1565259004634976 x2_input_acceleration
   + 0.5556056818779933 x3_input_acceleration >= -0.8414811459996032
 CBF_Constraint_for_table: x3_input_acceleration >= -1.627825494959113
 Relaxed_CLF_constraint: - 0.0863344878592454 x1_input_acceleration
   + 0.0775084735849321 x2_input_acceleration
   + 0.2785670492557996 x3_input_acceleration + relaxation_CLF
   <= -20.03642432328099
 CBF_Constraint_obs2: - 0.2939176124210404 x1_input_acceleration
   - 0.1595259004599685 x2_input_acceleration
   + 0.5582503254610696 x3_input_acceleration >= -0.4793493771077985
 CBF_Constraint_obs3: 0.0860823875789596 x1_input_acceleration
   - 0.1595259004599685 x2_input_acceleration
   + 0.5582503254610696 x3_input_acceleration >= -0.3806276453078101
 CBF_Constraint_obs1: - 0.6139176124210404 x1_input_acceleration
   - 0.1595259004599685 x2_input_acceleration
   + 0.5582503254610696 x3_input_acceleration >= -0.8424834670446307
 CBF_Constraint_for_table: x3_input_acceleration >= -1.634437103916804
 Relaxed_CLF_constraint: - 0.0850779125795388 x1_input_acceleration
   + 0.0760084735866966 x2_input_acceleration
   + 0.2798893710473378 x3_input_acceleration + relaxation_CLF
   <= -20.03654143971646
 CBF_Constraint_obs2: - 0.291318885192252 x1_input_acceleration
   - 0.1625717960347473 x2_input_acceleration
   + 0.5609174995976232 x3_input_acceleration >= -0.4824059038630869
 CBF_Constraint_obs3: 0.0886672865301601 x1_input_acceleration
   - 0.1625259004582702 x2_input_acceleration
   + 0.5608949690418981 x3_input_acceleration >= -0.3860996528961311
 CBF_Constraint_obs1: - 0.6113327134698399 x1_input_acceleration
   - 0.1625259004582702 x2_input_acceleration
   + 0.5608949690418981 x3_input_acceleration >= -0.8434319014683509
 CBF_Constraint_for_table: x3_input_acceleration >= -1.641460878987497
 Relaxed_CLF_constraint: - 0.0837854631039386 x1_input_acceleration
   + 0.0745084735875458 x2_input_acceleration
   + 0.281211692837752 x3_input_acceleration + relaxation_CLF
   <= -20.03666061306102
 CBF_Constraint_obs2: - 0.2886605773328862 x1_input_acceleration
   - 0.1655717960347196 x2_input_acceleration
   + 0.5635621431766413 x3_input_acceleration >= -0.4854265472127089
 CBF_Constraint_obs3: 0.0913394226671138 x1_input_acceleration
   - 0.1655717960347196 x2_input_acceleration
   + 0.5635621431766413 x3_input_acceleration >= -0.3916989987464671
 CBF_Constraint_obs1: - 0.6086605773328863 x1_input_acceleration
   - 0.1655717960347196 x2_input_acceleration
   + 0.5635621431766413 x3_input_acceleration >= -0.8443550090790179
 CBF_Constraint_for_table: x3_input_acceleration >= -1.648072487935042
 Relaxed_CLF_constraint: - 0.0824563091742557 x1_input_acceleration
   + 0.0730084735875596 x2_input_acceleration
   + 0.2825340146272611 x3_input_acceleration + relaxation_CLF
   <= -20.03678189982385
 CBF_Constraint_obs2: - 0.2859271093740389 x1_input_acceleration
   - 0.1685717960343471 x2_input_acceleration
   + 0.5662067867553457 x3_input_acceleration >= -0.4884511095763158
 CBF_Constraint_obs3: 0.0940728906259611 x1_input_acceleration
   - 0.1685717960343471 x2_input_acceleration
   + 0.5662067867553457 x3_input_acceleration >= -0.397320355670979
 CBF_Constraint_obs1: - 0.6059271093740389 x1_input_acceleration
   - 0.1685717960343471 x2_input_acceleration
   + 0.5662067867553457 x3_input_acceleration >= -0.845192797075547
 CBF_Constraint_for_table: x3_input_acceleration >= -1.654684096881803
 Relaxed_CLF_constraint: - 0.081089575194832 x1_input_acceleration
   + 0.0715084735877458 x2_input_acceleration
   + 0.2838563364166133 x3_input_acceleration + relaxation_CLF
   <= -20.03690536028959
 CBF_Constraint_obs2: - 0.283236795588614 x1_input_acceleration
   - 0.171870892120481 x2_input_acceleration
   + 0.56853492969745 x3_input_acceleration >= -0.491243315274601
 CBF_Constraint_obs3: 0.096763204411386 x1_input_acceleration
   - 0.171870892120481 x2_input_acceleration
   + 0.56853492969745 x3_input_acceleration >= -0.4026683594654176
 CBF_Constraint_obs1: - 0.603236795588614 x1_input_acceleration
   - 0.171870892120481 x2_input_acceleration
   + 0.56853492969745 x3_input_acceleration >= -0.8458327517454921
 CBF_Constraint_for_table: x3_input_acceleration >= -1.660912448030551
 Relaxed_CLF_constraint: - 0.079684336820186 x1_input_acceleration
   + 0.0700084735984203 x2_input_acceleration
   + 0.2851786582195441 x3_input_acceleration + relaxation_CLF
   <= -20.03703105880558
 CBF_Constraint_obs2: - 0.2803516484405887 x1_input_acceleration
   - 0.1748589711914326 x2_input_acceleration
   + 0.5711775467178259 x3_input_acceleration >= -0.4942706205800948
 CBF_Constraint_obs3: 0.0996483515594113 x1_input_acceleration
   - 0.1748589711914326 x2_input_acceleration
   + 0.5711775467178259 x3_input_acceleration >= -0.4084365545615357
 CBF_Constraint_obs1: - 0.6003516484405887 x1_input_acceleration
   - 0.1748589711914326 x2_input_acceleration
   + 0.5711775467178259 x3_input_acceleration >= -0.8465519393325659
 CBF_Constraint_for_table: x3_input_acceleration >= -1.667524056976297
 Relaxed_CLF_constraint: - 0.0782396174789614 x1_input_acceleration
   + 0.068508473598467 x2_input_acceleration
   + 0.2865009800086932 x3_input_acceleration + relaxation_CLF
   <= -20.03715906409759
 CBF_Constraint_obs2: - 0.27728676770545 x1_input_acceleration
   - 0.177837394286162 x2_input_acceleration
   + 0.5738405485477898 x3_input_acceleration >= -0.4972588318858204
 CBF_Constraint_obs3: 0.10271323229455 x1_input_acceleration
   - 0.177837394286162 x2_input_acceleration
   + 0.5738405485477898 x3_input_acceleration >= -0.4143364025656429
 CBF_Constraint_obs1: - 0.5972867677054501 x1_input_acceleration
   - 0.177837394286162 x2_input_acceleration
   + 0.5738405485477898 x3_input_acceleration >= -0.8470882460501805
 CBF_Constraint_for_table: x3_input_acceleration >= -1.674175601086729
 Relaxed_CLF_constraint: - 0.0767543839900542 x1_input_acceleration
   + 0.067008473610398 x2_input_acceleration
   + 0.2878233018083797 x3_input_acceleration + relaxation_CLF
   <= -20.03728944962447
 CBF_Constraint_obs2: - 0.274286767746635 x1_input_acceleration
   - 0.1808373931864233 x2_input_acceleration
   + 0.5764851921402121 x3_input_acceleration >= -0.5003382131153684
 CBF_Constraint_obs3: 0.1057132322533651 x1_input_acceleration
   - 0.1808373931864233 x2_input_acceleration
   + 0.5764851921402121 x3_input_acceleration >= -0.4202657837560653
 CBF_Constraint_obs1: - 0.5942867677466349 x1_input_acceleration
   - 0.1808373931864233 x2_input_acceleration
   + 0.5764851921402121 x3_input_acceleration >= -0.8477676273126764
 CBF_Constraint_for_table: x3_input_acceleration >= -1.680787210067785
 Relaxed_CLF_constraint: - 0.0752543840106466 x1_input_acceleration
   + 0.0655084741602673 x2_input_acceleration
   + 0.2891456236045909 x3_input_acceleration + relaxation_CLF
   <= -20.03742390965973
 CBF_Constraint_obs2: - 0.2712967813352186 x1_input_acceleration
   - 0.1835135014359583 x2_input_acceleration
   + 0.5793793407628116 x3_input_acceleration >= -0.5036994615936063
 CBF_Constraint_obs3: 0.1087032186647814 x1_input_acceleration
   - 0.1835135014359583 x2_input_acceleration
   + 0.5793793407628116 x3_input_acceleration >= -0.4264675193251487
 CBF_Constraint_obs1: - 0.5912867677548964 x1_input_acceleration
   - 0.1838373930756678 x2_input_acceleration
   + 0.5791298357197818 x3_input_acceleration >= -0.8485094942946397
 CBF_Constraint_for_table: x3_input_acceleration >= -1.687318054723038
 Relaxed_CLF_constraint: - 0.0737543840147774 x1_input_acceleration
   + 0.064008474215645 x2_input_acceleration
   + 0.2904679453943758 x3_input_acceleration + relaxation_CLF
   <= -20.03756336849378
 CBF_Constraint_obs2: - 0.2683673532991742 x1_input_acceleration
   - 0.1868888911480988 x2_input_acceleration
   + 0.5817318023864557 x3_input_acceleration >= -0.5067004798239551
 CBF_Constraint_obs3: 0.1116326467008259 x1_input_acceleration
   - 0.1868888911480988 x2_input_acceleration
   + 0.5817318023864557 x3_input_acceleration >= -0.4322514941897396
 CBF_Constraint_obs1: - 0.58829678139976 x1_input_acceleration
   - 0.1865135010953034 x2_input_acceleration
   + 0.5820239843551447 x3_input_acceleration >= -0.849540384325978
 CBF_Constraint_for_table: x3_input_acceleration >= -1.694026819274855
 Relaxed_CLF_constraint: - 0.0722543840470481 x1_input_acceleration
   + 0.0625084743859725 x2_input_acceleration
   + 0.2917902671905424 x3_input_acceleration + relaxation_CLF
   <= -20.03770782616448
 CBF_Constraint_obs2: - 0.2653673533226322 x1_input_acceleration
   - 0.1898888908924203 x2_input_acceleration
   + 0.5843764459681768 x3_input_acceleration >= -0.5099668171202622
 CBF_Constraint_obs3: 0.1146326466773678 x1_input_acceleration
   - 0.1898888908924203 x2_input_acceleration
   + 0.5843764459681768 x3_input_acceleration >= -0.4383678314637617
 CBF_Constraint_obs1: - 0.5853673533226322 x1_input_acceleration
   - 0.1898888908924203 x2_input_acceleration
   + 0.5843764459681768 x3_input_acceleration >= -0.8502606997783679
 CBF_Constraint_for_table: x3_input_acceleration >= -1.700638428229158
 Relaxed_CLF_constraint: - 0.0707543840587771 x1_input_acceleration
   + 0.0610084745138118 x2_input_acceleration
   + 0.2931125889814029 x3_input_acceleration + relaxation_CLF
   <= -20.03785728265817
 CBF_Constraint_obs2: - 0.2623621081182814 x1_input_acceleration
   - 0.1928614726781355 x2_input_acceleration
   + 0.5870419511729206 x3_input_acceleration >= -0.5133095948209471
 CBF_Constraint_obs3: 0.1176326466729784 x1_input_acceleration
   - 0.1928888908147322 x2_input_acceleration
   + 0.5870210895472492 x3_input_acceleration >= -0.4445466541703337
 CBF_Constraint_obs1: - 0.5823673533270216 x1_input_acceleration
   - 0.1928888908147322 x2_input_acceleration
   + 0.5870210895472492 x3_input_acceleration >= -0.8511895224926216
 CBF_Constraint_for_table: x3_input_acceleration >= -1.707598724348775
 Relaxed_CLF_constraint: - 0.0692543840609718 x1_input_acceleration
   + 0.0595084745526558 x2_input_acceleration
   + 0.294434910770939 x3_input_acceleration + relaxation_CLF
   <= -20.03801173797438
 CBF_Constraint_obs2: - 0.2593621081185744 x1_input_acceleration
   - 0.1958614726766117 x2_input_acceleration
   + 0.5896865947511793 x3_input_acceleration >= -0.5167008745450047
 CBF_Constraint_obs3: 0.1206378918814256 x1_input_acceleration
   - 0.1958614726766117 x2_input_acceleration
   + 0.5896865947511793 x3_input_acceleration >= -0.4508068718323591
 CBF_Constraint_obs1: - 0.5793621081185745 x1_input_acceleration
   - 0.1958614726766117 x2_input_acceleration
   + 0.5896865947511793 x3_input_acceleration >= -0.8521905610398643
 CBF_Constraint_for_table: x3_input_acceleration >= -1.714210333294422
 Relaxed_CLF_constraint: - 0.0677543840611183 x1_input_acceleration
   + 0.0580084745534177 x2_input_acceleration
   + 0.2957572325600684 x3_input_acceleration + relaxation_CLF
   <= -20.03817119211644
 CBF_Constraint_obs2: - 0.2563621081186375 x1_input_acceleration
   - 0.1988614726750044 x2_input_acceleration
   + 0.5923312383294117 x3_input_acceleration >= -0.5201546396179473
 CBF_Constraint_obs3: 0.1236378918813625 x1_input_acceleration
   - 0.1988614726750044 x2_input_acceleration
   + 0.5923312383294117 x3_input_acceleration >= -0.4571106369052416
 CBF_Constraint_obs1: - 0.5763621081186374 x1_input_acceleration
   - 0.1988614726750044 x2_input_acceleration
   + 0.5923312383294117 x3_input_acceleration >= -0.8532443261128571
 CBF_Constraint_for_table: x3_input_acceleration >= -1.720396067053083
 Relaxed_CLF_constraint: - 0.0662543840611499 x1_input_acceleration
   + 0.0565084745542214 x2_input_acceleration
   + 0.2970795543491846 x3_input_acceleration + relaxation_CLF
   <= -20.03833564508643
 CBF_Constraint_obs2: - 0.2533559092361903 x1_input_acceleration
   - 0.2018832879709931 x2_input_acceleration
   + 0.5949587157700538 x3_input_acceleration >= -0.5236524402220339
 CBF_Constraint_obs3: 0.1266440907638097 x1_input_acceleration
   - 0.2018832879709931 x2_input_acceleration
   + 0.5949587157700538 x3_input_acceleration >= -0.4634643264476532
 CBF_Constraint_obs1: - 0.5733559092361903 x1_input_acceleration
   - 0.2018832879709931 x2_input_acceleration
   + 0.5949587157700538 x3_input_acceleration >= -0.8543371676109862
 CBF_Constraint_for_table: x3_input_acceleration >= -1.727013636463403
 Relaxed_CLF_constraint: - 0.0647543840614546 x1_input_acceleration
   + 0.0550084745562209 x2_input_acceleration
   + 0.2984018761383533 x3_input_acceleration + relaxation_CLF
   <= -20.03850509688443
 CBF_Constraint_obs2: - 0.2503559092440256 x1_input_acceleration
   - 0.204883287804339 x2_input_acceleration
   + 0.5976033593497359 x3_input_acceleration >= -0.5272312725262598
 CBF_Constraint_obs3: 0.1296440907559744 x1_input_acceleration
   - 0.204883287804339 x2_input_acceleration
   + 0.5976033593497359 x3_input_acceleration >= -0.4698931587444355
 CBF_Constraint_obs1: - 0.5703559092440256 x1_input_acceleration
   - 0.204883287804339 x2_input_acceleration
   + 0.5976033593497359 x3_input_acceleration >= -0.855516
 CBF_Constraint_for_table: x3_input_acceleration >= -1.733625245412609
 Relaxed_CLF_constraint: - 0.0632543840653722 x1_input_acceleration
   + 0.053508474639548 x2_input_acceleration
   + 0.2997241979281944 x3_input_acceleration + relaxation_CLF
   <= -20.03867954751417
 CBF_Constraint_obs2: - 0.2473559092818748 x1_input_acceleration
   - 0.2078832875046333 x2_input_acceleration
   + 0.6002480029321857 x3_input_acceleration >= -0.5308725901305928
 CBF_Constraint_obs3: 0.1326440907181252 x1_input_acceleration
   - 0.2078832875046333 x2_input_acceleration
   + 0.6002480029321857 x3_input_acceleration >= -0.4763844763128118
 CBF_Constraint_obs1: - 0.5673559092818748 x1_input_acceleration
   - 0.2078832875046333 x2_input_acceleration
   + 0.6002480029321857 x3_input_acceleration >= -0.8567573175560927
 CBF_Constraint_for_table: x3_input_acceleration >= -1.740624880606221
 Relaxed_CLF_constraint: - 0.0617543840842969 x1_input_acceleration
   + 0.0520084747894008 x2_input_acceleration
   + 0.3010465197194193 x3_input_acceleration + relaxation_CLF
   <= -20.03885899697549
 CBF_Constraint_obs2: - 0.2443787974800732 x1_input_acceleration
   - 0.2108695782432517 x2_input_acceleration
   + 0.6029126736738053 x3_input_acceleration >= -0.5346133347034875
 CBF_Constraint_obs3: 0.1356212025199268 x1_input_acceleration
   - 0.2108695782432517 x2_input_acceleration
   + 0.6029126736738053 x3_input_acceleration >= -0.4829534770974178
 CBF_Constraint_obs1: - 0.5643787974800731 x1_input_acceleration
   - 0.2108695782432517 x2_input_acceleration
   + 0.6029126736738053 x3_input_acceleration >= -0.858116372687546
 CBF_Constraint_for_table: x3_input_acceleration >= -1.747236489558659
 Relaxed_CLF_constraint: - 0.0602543840915992 x1_input_acceleration
   + 0.0505084748859425 x2_input_acceleration
   + 0.3023688415099068 x3_input_acceleration + relaxation_CLF
   <= -20.03904344526144
 CBF_Constraint_obs2: - 0.241300596194942 x1_input_acceleration
   - 0.2135156454524532 x2_input_acceleration
   + 0.6058413929919559 x3_input_acceleration >= -0.5385733775493072
 CBF_Constraint_obs3: 0.138699403805058 x1_input_acceleration
   - 0.2135156454524532 x2_input_acceleration
   + 0.6058413929919559 x3_input_acceleration >= -0.4898378111641122
 CBF_Constraint_obs1: - 0.561300596194942 x1_input_acceleration
   - 0.2135156454524532 x2_input_acceleration
   + 0.6058413929919559 x3_input_acceleration >= -0.8596138545052606
 CBF_Constraint_for_table: x3_input_acceleration >= -1.753509246105988
 Relaxed_CLF_constraint: - 0.0587543840960063 x1_input_acceleration
   + 0.0490084750910036 x2_input_acceleration
   + 0.3036911633004823 x3_input_acceleration + relaxation_CLF
   <= -20.03923289237922
 CBF_Constraint_obs2: - 0.2382104739797604 x1_input_acceleration
   - 0.2164755908039895 x2_input_acceleration
   + 0.608492235455808 x3_input_acceleration >= -0.5423358899220061
 CBF_Constraint_obs3: 0.1417895260202396 x1_input_acceleration
   - 0.2164755908039895 x2_input_acceleration
   + 0.608492235455808 x3_input_acceleration >= -0.4965359396412338
 CBF_Constraint_obs1: - 0.5582104739797604 x1_input_acceleration
   - 0.2164755908039895 x2_input_acceleration
   + 0.608492235455808 x3_input_acceleration >= -0.8609042691058144
 CBF_Constraint_for_table: x3_input_acceleration >= -1.760525570596002
 Relaxed_CLF_constraint: - 0.0572543840998656 x1_input_acceleration
   + 0.0475084752545909 x2_input_acceleration
   + 0.3050134850908801 x3_input_acceleration + relaxation_CLF
   <= -20.03942733832278
 CBF_Constraint_obs2: - 0.2352104739953529 x1_input_acceleration
   - 0.2194755904355092 x2_input_acceleration
   + 0.6111368790387171 x3_input_acceleration >= -0.5462272334941467
 CBF_Constraint_obs3: 0.1447895260046471 x1_input_acceleration
   - 0.2194755904355092 x2_input_acceleration
   + 0.6111368790387171 x3_input_acceleration >= -0.5032772831985616
 CBF_Constraint_obs1: - 0.5552104739953528 x1_input_acceleration
   - 0.2194755904355092 x2_input_acceleration
   + 0.6111368790387171 x3_input_acceleration >= -0.8623956126904289
 CBF_Constraint_for_table: x3_input_acceleration >= -1.767137179553274
 Relaxed_CLF_constraint: - 0.0557543841076619 x1_input_acceleration
   + 0.046008475438831 x2_input_acceleration
   + 0.3063358068823346 x3_input_acceleration + relaxation_CLF
   <= -20.03962678309507
 CBF_Constraint_obs2: - 0.2322834300805837 x1_input_acceleration
   - 0.222521366798984 x2_input_acceleration
   + 0.6137672175021975 x3_input_acceleration >= -0.550226934815599
 CBF_Constraint_obs3: 0.1477165699194163 x1_input_acceleration
   - 0.222521366798984 x2_input_acceleration
   + 0.6137672175021975 x3_input_acceleration >= -0.5100576762390445
 CBF_Constraint_obs1: - 0.5522834300805837 x1_input_acceleration
   - 0.222521366798984 x2_input_acceleration
   + 0.6137672175021975 x3_input_acceleration >= -0.8640536788800659
 CBF_Constraint_for_table: x3_input_acceleration >= -1.773687693737946
 Relaxed_CLF_constraint: - 0.0542543841076747 x1_input_acceleration
   + 0.0445084754406874 x2_input_acceleration
   + 0.3076581286714479 x3_input_acceleration + relaxation_CLF
   <= -20.03983122668743
 CBF_Constraint_obs2: - 0.229301549932267 x1_input_acceleration
   - 0.2258234423997862 x2_input_acceleration
   + 0.6161815487404161 x3_input_acceleration >= -0.554068911183459
 CBF_Constraint_obs3: 0.150698450067733 x1_input_acceleration
   - 0.2258234423997862 x2_input_acceleration
   + 0.6161815487404161 x3_input_acceleration >= -0.5167324387478055
 CBF_Constraint_obs1: - 0.549301549932267 x1_input_acceleration
   - 0.2258234423997862 x2_input_acceleration
   + 0.6161815487404161 x3_input_acceleration >= -0.8655101511292727
 CBF_Constraint_for_table: x3_input_acceleration >= -1.779894885187222
 Relaxed_CLF_constraint: - 0.0527543841275105 x1_input_acceleration
   + 0.0430084758101471 x2_input_acceleration
   + 0.3089804504642633 x3_input_acceleration + relaxation_CLF
   <= -20.04004066912211
 CBF_Constraint_obs2: - 0.2263730755835567 x1_input_acceleration
   - 0.2288633772960398 x2_input_acceleration
   + 0.6188118872234634 x3_input_acceleration >= -0.5581892082319656
 CBF_Constraint_obs3: 0.1536269244164433 x1_input_acceleration
   - 0.2288633772960398 x2_input_acceleration
   + 0.6188118872234634 x3_input_acceleration >= -0.5236347864275868
 CBF_Constraint_obs1: - 0.5463730755835567 x1_input_acceleration
   - 0.2288633772960398 x2_input_acceleration
   + 0.6188118872234634 x3_input_acceleration >= -0.8672876686988109
 CBF_Constraint_for_table: x3_input_acceleration >= -1.786988397734714
 Relaxed_CLF_constraint: - 0.0512543841662901 x1_input_acceleration
   + 0.0415084759180201 x2_input_acceleration
   + 0.3103027722631599 x3_input_acceleration + relaxation_CLF
   <= -20.0402551103774
 CBF_Constraint_obs2: - 0.2233730755840031 x1_input_acceleration
   - 0.2318633772941923 x2_input_acceleration
   + 0.621456530801751 x3_input_acceleration >= -0.5623304703778418
 CBF_Constraint_obs3: 0.1566269244159969 x1_input_acceleration
   - 0.2318633772941923 x2_input_acceleration
   + 0.621456530801751 x3_input_acceleration >= -0.5306260485730387
 CBF_Constraint_obs1: - 0.5433730755840032 x1_input_acceleration
   - 0.2318633772941923 x2_input_acceleration
   + 0.621456530801751 x3_input_acceleration >= -0.8690289308450444
 CBF_Constraint_for_table: x3_input_acceleration >= -1.793600006680434
 Relaxed_CLF_constraint: - 0.0497543841665134 x1_input_acceleration
   + 0.0400084759189439 x2_input_acceleration
   + 0.3116250940523038 x3_input_acceleration + relaxation_CLF
   <= -20.0404745504529
 CBF_Constraint_obs2: - 0.2203001195384223 x1_input_acceleration
   - 0.2348326204666469 x2_input_acceleration
   + 0.6241167908044616 x3_input_acceleration >= -0.5665003385849193
 CBF_Constraint_obs3: 0.1596269243763726 x1_input_acceleration
   - 0.234863376463351 x2_input_acceleration
   + 0.6241011743875304 x3_input_acceleration >= -0.5376797955769007
 CBF_Constraint_obs1: - 0.5403730756236274 x1_input_acceleration
   - 0.234863376463351 x2_input_acceleration
   + 0.6241011743875304 x3_input_acceleration >= -0.8708326779182486
 CBF_Constraint_for_table: x3_input_acceleration >= -1.800211615644882
 Relaxed_CLF_constraint: - 0.0482543841863255 x1_input_acceleration
   + 0.0385084763343645 x2_input_acceleration
   + 0.3129474158451935 x3_input_acceleration + relaxation_CLF
   <= -20.04069898937079
 CBF_Constraint_obs2: - 0.2173001195632097 x1_input_acceleration
   - 0.2378326196543394 x2_input_acceleration
   + 0.6267614343917396 x3_input_acceleration >= -0.5707669907182772
 CBF_Constraint_obs3: 0.1626998804367903 x1_input_acceleration
   - 0.2378326196543394 x2_input_acceleration
   + 0.6267614343917396 x3_input_acceleration >= -0.5448318771332281
 CBF_Constraint_obs1: - 0.5373001195632097 x1_input_acceleration
   - 0.2378326196543394 x2_input_acceleration
   + 0.6267614343917396 x3_input_acceleration >= -0.872607086368845
 CBF_Constraint_for_table: x3_input_acceleration >= -1.806394071170694
 Relaxed_CLF_constraint: - 0.0467543841987192 x1_input_acceleration
   + 0.0370084767405182 x2_input_acceleration
   + 0.3142697376388325 x3_input_acceleration + relaxation_CLF
   <= -20.04092842711522
 CBF_Constraint_obs2: - 0.2143592473817374 x1_input_acceleration
   - 0.2408671898895494 x2_input_acceleration
   + 0.6293555332339453 x3_input_acceleration >= -0.575069096644421
 CBF_Constraint_obs3: 0.1656998804258797 x1_input_acceleration
   - 0.2408326191955796 x2_input_acceleration
   + 0.6294060779727149 x3_input_acceleration >= -0.5520110147884363
 CBF_Constraint_obs1: - 0.5343001195741203 x1_input_acceleration
   - 0.2408326191955796 x2_input_acceleration
   + 0.6294060779727149 x3_input_acceleration >= -0.8745362240431469
 CBF_Constraint_for_table: x3_input_acceleration >= -1.813334995785516
 Relaxed_CLF_constraint: - 0.0452543842041745 x1_input_acceleration
   + 0.0355084769698981 x2_input_acceleration
   + 0.3155920594293201 x3_input_acceleration + relaxation_CLF
   <= -20.0411628636805
 CBF_Constraint_obs2: - 0.2113854734442015 x1_input_acceleration
   - 0.2438685004524834 x2_input_acceleration
   + 0.6319991039345914 x3_input_acceleration >= -0.5794731640878731
 CBF_Constraint_obs3: 0.1686145265557985 x1_input_acceleration
   - 0.2438685004524834 x2_input_acceleration
   + 0.6319991039345914 x3_input_acceleration >= -0.5591569643158817
 CBF_Constraint_obs1: - 0.5313854734442015 x1_input_acceleration
   - 0.2438685004524834 x2_input_acceleration
   + 0.6319991039345914 x3_input_acceleration >= -0.8765815428432344
 CBF_Constraint_for_table: x3_input_acceleration >= -1.820069986360831
 Relaxed_CLF_constraint: - 0.0437543842135559 x1_input_acceleration
   + 0.0340084773395236 x2_input_acceleration
   + 0.3169143812214461 x3_input_acceleration + relaxation_CLF
   <= -20.04140229907753
 CBF_Constraint_obs2: - 0.2083854734442299 x1_input_acceleration
   - 0.2468685004501903 x2_input_acceleration
   + 0.6346437475128157 x3_input_acceleration >= -0.583926560392765
 CBF_Constraint_obs3: 0.1716145265557701 x1_input_acceleration
   - 0.2468685004501903 x2_input_acceleration
   + 0.6346437475128157 x3_input_acceleration >= -0.5664603606207466
 CBF_Constraint_obs1: - 0.5283854734442299 x1_input_acceleration
   - 0.2468685004501903 x2_input_acceleration
   + 0.6346437475128157 x3_input_acceleration >= -0.8786349391481491
 CBF_Constraint_for_table: x3_input_acceleration >= -1.826681595306392
 Relaxed_CLF_constraint: - 0.0422543842135701 x1_input_acceleration
   + 0.0325084773406702 x2_input_acceleration
   + 0.3182367030105583 x3_input_acceleration + relaxation_CLF
   <= -20.04164673329092
 CBF_Constraint_obs2: - 0.20533016033457 x1_input_acceleration
   - 0.2499236943456495 x2_input_acceleration
   + 0.6373057956473487 x3_input_acceleration >= -0.5884762561301226
 CBF_Constraint_obs3: 0.17466983966543 x1_input_acceleration
   - 0.2499236943456495 x2_input_acceleration
   + 0.6373057956473487 x3_input_acceleration >= -0.573912603812281
 CBF_Constraint_obs1: - 0.52533016033457 x1_input_acceleration
   - 0.2499236943456495 x2_input_acceleration
   + 0.6373057956473487 x3_input_acceleration >= -0.8807403843977785
 CBF_Constraint_for_table: x3_input_acceleration >= -1.832884316388879
 Relaxed_CLF_constraint: - 0.0407543842139159 x1_input_acceleration
   + 0.0310084773434716 x2_input_acceleration
   + 0.3195590247996875 x3_input_acceleration + relaxation_CLF
   <= -20.0418961663323
 CBF_Constraint_obs2: - 0.2023301604412477 x1_input_acceleration
   - 0.2529236863371795 x2_input_acceleration
   + 0.6399504392344315 x3_input_acceleration >= -0.5930555620136385
 CBF_Constraint_obs3: 0.1776698395587523 x1_input_acceleration
   - 0.2529236863371795 x2_input_acceleration
   + 0.6399504392344315 x3_input_acceleration >= -0.5813419095944532
 CBF_Constraint_obs1: - 0.5223301604412477 x1_input_acceleration
   - 0.2529236863371795 x2_input_acceleration
   + 0.6399504392344315 x3_input_acceleration >= -0.8829196903666368
 CBF_Constraint_for_table: x3_input_acceleration >= -1.839495925356587
 Relaxed_CLF_constraint: - 0.0392543842672548 x1_input_acceleration
   + 0.0295084813477066 x2_input_acceleration
   + 0.3208813465932289 x3_input_acceleration + relaxation_CLF
   <= -20.04215059829885
 CBF_Constraint_obs2: - 0.1992810462394441 x1_input_acceleration
   - 0.2555233813905118 x2_input_acceleration
   + 0.6428947749702576 x3_input_acceleration >= -0.5978985318894832
 CBF_Constraint_obs3: 0.1807189537605559 x1_input_acceleration
   - 0.2555233813905118 x2_input_acceleration
   + 0.6428947749702576 x3_input_acceleration >= -0.5890815379620112
 CBF_Constraint_obs1: - 0.5193301604667391 x1_input_acceleration
   - 0.2559236861848233 x2_input_acceleration
   + 0.6425950828163269 x3_input_acceleration >= -0.8851614864756986
 CBF_Constraint_for_table: x3_input_acceleration >= -1.846444896600754
 Relaxed_CLF_constraint: - 0.0377543842800004 x1_input_acceleration
   + 0.0280084814238847 x2_input_acceleration
   + 0.3222036683841766 x3_input_acceleration + relaxation_CLF
   <= -20.0424100289937
 CBF_Constraint_obs2: - 0.1962810462408852 x1_input_acceleration
   - 0.2585233813759913 x2_input_acceleration
   + 0.6455394185486347 x3_input_acceleration >= -0.6026021609145027
 CBF_Constraint_obs3: 0.1837189537591148 x1_input_acceleration
   - 0.2585233813759913 x2_input_acceleration
   + 0.6455394185486347 x3_input_acceleration >= -0.5966351669856618
 CBF_Constraint_obs1: - 0.5162810462408851 x1_input_acceleration
   - 0.2585233813759913 x2_input_acceleration
   + 0.6455394185486347 x3_input_acceleration >= -0.8876269979072107
 CBF_Constraint_for_table: x3_input_acceleration >= -1.852630332336553
 Relaxed_CLF_constraint: - 0.036254384280721 x1_input_acceleration
   + 0.0265084814311449 x2_input_acceleration
   + 0.3235259901733653 x3_input_acceleration + relaxation_CLF
   <= -20.04267445851401
 CBF_Constraint_obs2: - 0.1933654464182059 x1_input_acceleration
   - 0.2618925725446024 x2_input_acceleration
   + 0.6478400241172384 x3_input_acceleration >= -0.6070932653384574
 CBF_Constraint_obs3: 0.1866345535817941 x1_input_acceleration
   - 0.2618925725446024 x2_input_acceleration
   + 0.6478400241172384 x3_input_acceleration >= -0.6038960912411617
 CBF_Constraint_obs1: - 0.5133654464182059 x1_input_acceleration
   - 0.2618925725446024 x2_input_acceleration
   + 0.6478400241172384 x3_input_acceleration >= -0.8897856224730222
 CBF_Constraint_for_table: x3_input_acceleration >= -1.859241941282171
 Relaxed_CLF_constraint: - 0.0347543842808804 x1_input_acceleration
   + 0.0250084814317088 x2_input_acceleration
   + 0.3248483119624888 x3_input_acceleration + relaxation_CLF
   <= -20.04294388686206
 CBF_Constraint_obs2: - 0.1902915366710144 x1_input_acceleration
   - 0.2648335587913104 x2_input_acceleration
   + 0.6505523785806342 x3_input_acceleration >= -0.6119576000845091
 CBF_Constraint_obs3: 0.1897084633289856 x1_input_acceleration
   - 0.2648335587913104 x2_input_acceleration
   + 0.6505523785806342 x3_input_acceleration >= -0.6116806402470454
 CBF_Constraint_obs1: - 0.5102915366710145 x1_input_acceleration
   - 0.2648335587913104 x2_input_acceleration
   + 0.6505523785806342 x3_input_acceleration >= -0.8921908294213208
 CBF_Constraint_for_table: x3_input_acceleration >= -1.866303267294329
 Relaxed_CLF_constraint: - 0.0332543842870454 x1_input_acceleration
   + 0.0235084840091909 x2_input_acceleration
   + 0.3261706337559543 x3_input_acceleration + relaxation_CLF
   <= -20.04321831408779
 CBF_Constraint_obs2: - 0.1872915366815415 x1_input_acceleration
   - 0.2678335574105144 x2_input_acceleration
   + 0.6531970221607479 x3_input_acceleration >= -0.6168491049342163
 CBF_Constraint_obs3: 0.1927084633184585 x1_input_acceleration
   - 0.2678335574105144 x2_input_acceleration
   + 0.6531970221607479 x3_input_acceleration >= -0.619422145086752
 CBF_Constraint_obs1: - 0.5072915366815415 x1_input_acceleration
   - 0.2678335574105144 x2_input_acceleration
   + 0.6531970221607479 x3_input_acceleration >= -0.8946823342794495
 CBF_Constraint_for_table: x3_input_acceleration >= -1.872914876244613
 Relaxed_CLF_constraint: - 0.0317543842923089 x1_input_acceleration
   + 0.0220084846995889 x2_input_acceleration
   + 0.3274929555460112 x3_input_acceleration + relaxation_CLF
   <= -20.04349774010113
 CBF_Constraint_obs2: - 0.184266264841943 x1_input_acceleration
   - 0.27056247155595 x2_input_acceleration
   + 0.6561371855964337 x3_input_acceleration >= -0.6220926421195928
 CBF_Constraint_obs3: 0.1956440897723148 x1_input_acceleration
   - 0.2708671705000418 x2_input_acceleration
   + 0.655804353260008 x3_input_acceleration >= -0.627156229590392
 CBF_Constraint_obs1: - 0.5043559102276851 x1_input_acceleration
   - 0.2708671705000418 x2_input_acceleration
   + 0.655804353260008 x3_input_acceleration >= -0.8972790724888411
 CBF_Constraint_for_table: x3_input_acceleration >= -1.879088689145961
 Relaxed_CLF_constraint: - 0.030254384557202 x1_input_acceleration
   + 0.0205084866646519 x2_input_acceleration
   + 0.3288152773494559 x3_input_acceleration + relaxation_CLF
   <= -20.04378216497144
 CBF_Constraint_obs2: - 0.1812662652170035 x1_input_acceleration
   - 0.2735624691806471 x2_input_acceleration
   + 0.6587818292085414 x3_input_acceleration >= -0.6271092274316068
 CBF_Constraint_obs3: 0.1987337347829965 x1_input_acceleration
   - 0.2735624691806471 x2_input_acceleration
   + 0.6587818292085414 x3_input_acceleration >= -0.6354062754754533
 CBF_Constraint_obs1: - 0.5012662652170035 x1_input_acceleration
   - 0.2735624691806471 x2_input_acceleration
   + 0.6587818292085414 x3_input_acceleration >= -0.9001222396052094
 CBF_Constraint_for_table: x3_input_acceleration >= -1.88570029817623
 Relaxed_CLF_constraint: - 0.0287543847447323 x1_input_acceleration
   + 0.0190084878523034 x2_input_acceleration
   + 0.3301375991555097 x3_input_acceleration + relaxation_CLF
   <= -20.04407158865146
 CBF_Constraint_obs2: - 0.1783711695252135 x1_input_acceleration
   - 0.2768665715073053 x2_input_acceleration
   + 0.6610944749232586 x3_input_acceleration >= -0.6318965966375766
 CBF_Constraint_obs3: 0.2017337346495912 x1_input_acceleration
   - 0.2765624686096613 x2_input_acceleration
   + 0.6614264727946575 x3_input_acceleration >= -0.6433353470659041
 CBF_Constraint_obs1: - 0.4982662653504089 x1_input_acceleration
   - 0.2765624686096613 x2_input_acceleration
   + 0.6614264727946575 x3_input_acceleration >= -0.9028013114291196
 CBF_Constraint_for_table: x3_input_acceleration >= -1.89231190714152
 Relaxed_CLF_constraint: - 0.027254384811435 x1_input_acceleration
   + 0.0175084881377963 x2_input_acceleration
   + 0.3314599209485677 x3_input_acceleration + relaxation_CLF
   <= -20.04436601113742
 CBF_Constraint_obs2: - 0.1753711695310666 x1_input_acceleration
   - 0.2798665712697277 x2_input_acceleration
   + 0.6637391185018553 x3_input_acceleration >= -0.6370374528082344
 CBF_Constraint_obs3: 0.2046288304689334 x1_input_acceleration
   - 0.2798665712697277 x2_input_acceleration
   + 0.6637391185018553 x3_input_acceleration >= -0.6509348417537211
 CBF_Constraint_obs1: - 0.4953711695310666 x1_input_acceleration
   - 0.2798665712697277 x2_input_acceleration
   + 0.6637391185018553 x3_input_acceleration >= -0.9053343884330878
 CBF_Constraint_for_table: x3_input_acceleration >= -1.899359822087768
 Relaxed_CLF_constraint: - 0.0257543848143615 x1_input_acceleration
   + 0.0160084882565851 x2_input_acceleration
   + 0.3327822427378661 x3_input_acceleration + relaxation_CLF
   <= -20.04466543244603
 CBF_Constraint_obs2: - 0.1721761439241291 x1_input_acceleration
   - 0.2825325402866306 x2_input_acceleration
   + 0.6667250582913486 x3_input_acceleration >= -0.6424894552637515
 CBF_Constraint_obs3: 0.2078238560758709 x1_input_acceleration
   - 0.2825325402866306 x2_input_acceleration
   + 0.6667250582913486 x3_input_acceleration >= -0.6594221185358289
 CBF_Constraint_obs1: - 0.4921761439241291 x1_input_acceleration
   - 0.2825325402866306 x2_input_acceleration
   + 0.6667250582913486 x3_input_acceleration >= -0.9082303704030548
 CBF_Constraint_for_table: x3_input_acceleration >= -1.905528270537637
 Relaxed_CLF_constraint: - 0.0242543852097453 x1_input_acceleration
   + 0.014508491533473 x2_input_acceleration
   + 0.3341045645346208 x3_input_acceleration + relaxation_CLF
   <= -20.04496985262854
 CBF_Constraint_obs2: - 0.1693559146907513 x1_input_acceleration
   - 0.2859241307153297 x2_input_acceleration
   + 0.6690542741127345 x3_input_acceleration >= -0.6475845700932201
 CBF_Constraint_obs3: 0.2106440853092487 x1_input_acceleration
   - 0.2859241307153297 x2_input_acceleration
   + 0.6690542741127345 x3_input_acceleration >= -0.6671964511370063
 CBF_Constraint_obs1: - 0.4893559146907513 x1_input_acceleration
   - 0.2859241307153297 x2_input_acceleration
   + 0.6690542741127345 x3_input_acceleration >= -0.9110693018458214
 CBF_Constraint_for_table: x3_input_acceleration >= -1.912544893102729
 Relaxed_CLF_constraint: - 0.0227543867887351 x1_input_acceleration
   + 0.0130084975772106 x2_input_acceleration
   + 0.3354268863353895 x3_input_acceleration + relaxation_CLF
   <= -20.0452792716816
 CBF_Constraint_obs2: - 0.1663559146935192 x1_input_acceleration
   - 0.2889241306936234 x2_input_acceleration
   + 0.6716989176909611 x3_input_acceleration >= -0.6529135995972399
 CBF_Constraint_obs3: 0.2136440853064808 x1_input_acceleration
   - 0.2889241306936234 x2_input_acceleration
   + 0.6716989176909611 x3_input_acceleration >= -0.6753754806383967
 CBF_Constraint_obs1: - 0.4863559146935192 x1_input_acceleration
   - 0.2889241306936234 x2_input_acceleration
   + 0.6716989176909611 x3_input_acceleration >= -0.9139983313520552
 CBF_Constraint_for_table: x3_input_acceleration >= -1.919156502048295
 Relaxed_CLF_constraint: - 0.021254386790119 x1_input_acceleration
   + 0.0115084975880637 x2_input_acceleration
   + 0.3367492081245028 x3_input_acceleration + relaxation_CLF
   <= -20.04559368945882
 CBF_Constraint_obs2: - 0.163273898707414 x1_input_acceleration
   - 0.2915240642799348 x2_input_acceleration
   + 0.6746006957067512 x3_input_acceleration >= -0.6584134308768039
 CBF_Constraint_obs3: 0.216726101292586 x1_input_acceleration
   - 0.2915240642799348 x2_input_acceleration
   + 0.6746006957067512 x3_input_acceleration >= -0.6838032271047606
 CBF_Constraint_obs1: - 0.4833559146986249 x1_input_acceleration
   - 0.2919241306556672 x2_input_acceleration
   + 0.6743435612691902 x3_input_acceleration >= -0.9169898461980319
 CBF_Constraint_for_table: x3_input_acceleration >= -1.925392601731783
 Relaxed_CLF_constraint: - 0.0197543867926719 x1_input_acceleration
   + 0.0100084976070418 x2_input_acceleration
   + 0.3380715299136173 x3_input_acceleration + relaxation_CLF
   <= -20.04591310606402
 CBF_Constraint_obs2: - 0.160196651094177 x1_input_acceleration
   - 0.29448436752712 x2_input_acceleration
   + 0.6772659624920709 x3_input_acceleration >= -0.6638414921124358
 CBF_Constraint_obs3: 0.2197261012861941 x1_input_acceleration
   - 0.2945240642205404 x2_input_acceleration
   + 0.6772453392849787 x3_input_acceleration >= -0.692106541968302
 CBF_Constraint_obs1: - 0.4802738987138059 x1_input_acceleration
   - 0.2945240642205404 x2_input_acceleration
   + 0.6772453392849787 x3_input_acceleration >= -0.9200858647174625
 CBF_Constraint_for_table: x3_input_acceleration >= -1.931912419524397
 Relaxed_CLF_constraint: - 0.0182543867958679 x1_input_acceleration
   + 0.008508497636739 x2_input_acceleration
   + 0.339393851702731 x3_input_acceleration + relaxation_CLF
   <= -20.04623752149718
 CBF_Constraint_obs2: - 0.1573053699719846 x1_input_acceleration
   - 0.2978290005164284 x2_input_acceleration
   + 0.6796382128436853 x3_input_acceleration >= -0.6691939916599952
 CBF_Constraint_obs3: 0.2226946300280154 x1_input_acceleration
   - 0.2978290005164284 x2_input_acceleration
   + 0.6796382128436853 x3_input_acceleration >= -0.7002538901866098
 CBF_Constraint_obs1: - 0.4773053699719846 x1_input_acceleration
   - 0.2978290005164284 x2_input_acceleration
   + 0.6796382128436853 x3_input_acceleration >= -0.9230382876375829
 CBF_Constraint_for_table: x3_input_acceleration >= -1.938616415669394
 Relaxed_CLF_constraint: - 0.0167543867987365 x1_input_acceleration
   + 0.00700849817013049 x2_input_acceleration
   + 0.34071617349185 x3_input_acceleration + relaxation_CLF
   <= -20.04656693576101
 CBF_Constraint_obs2: - 0.1543053700013509 x1_input_acceleration
   - 0.300829 x2_input_acceleration
   + 0.6822828564219521 x3_input_acceleration >= -0.674772663241237
 CBF_Constraint_obs3: 0.2256946299986491 x1_input_acceleration
   - 0.300829 x2_input_acceleration
   + 0.6822828564219521 x3_input_acceleration >= -0.7086825617399537
 CBF_Constraint_obs1: - 0.4743053700013509 x1_input_acceleration
   - 0.300829 x2_input_acceleration
   + 0.6822828564219521 x3_input_acceleration >= -0.9262169592423177
 CBF_Constraint_for_table: x3_input_acceleration >= -1.945228024615061
 Relaxed_CLF_constraint: - 0.0152543868134196 x1_input_acceleration
   + 0.0055084984504189 x2_input_acceleration
   + 0.3420384952809834 x3_input_acceleration + relaxation_CLF
   <= -20.04690134885053
 CBF_Constraint_obs2: - 0.1512944027505411 x1_input_acceleration
   - 0.303845327565551 x2_input_acceleration
   + 0.6849211819078476 x3_input_acceleration >= -0.6804112560163784
 CBF_Constraint_obs3: 0.2287055972494589 x1_input_acceleration
   - 0.303845327565551 x2_input_acceleration
   + 0.6849211819078476 x3_input_acceleration >= -0.7171815734033643
 CBF_Constraint_obs1: - 0.4712944027505411 x1_input_acceleration
   - 0.303845327565551 x2_input_acceleration
   + 0.6849211819078476 x3_input_acceleration >= -0.9294467782168114
 CBF_Constraint_for_table: x3_input_acceleration >= -1.95181847391177
 Relaxed_CLF_constraint: - 0.0137543868153341 x1_input_acceleration
   + 0.00400850048190338 x2_input_acceleration
   + 0.3433608170701042 x3_input_acceleration + relaxation_CLF
   <= -20.04724076077273
 CBF_Constraint_obs2: - 0.1482944027594696 x1_input_acceleration
   - 0.3062592391592587 x2_input_acceleration
   + 0.6875658254862929 x3_input_acceleration >= -0.6856658947960538
 CBF_Constraint_obs3: 0.2317055972405304 x1_input_acceleration
   - 0.3062592391592587 x2_input_acceleration
   + 0.6875658254862929 x3_input_acceleration >= -0.7252862121745576
 CBF_Constraint_obs1: - 0.4682944027594697 x1_input_acceleration
   - 0.3062592391592587 x2_input_acceleration
   + 0.6875658254862929 x3_input_acceleration >= -0.9323014170036295
 CBF_Constraint_for_table: x3_input_acceleration >= -1.958430082857883
 Relaxed_CLF_constraint: - 0.0122543868197984 x1_input_acceleration
   + 0.00280154468504956 x2_input_acceleration
   + 0.3446831388593268 x3_input_acceleration + relaxation_CLF
   <= -20.04758579394515
 CBF_Constraint_obs2: - 0.1453535306073959 x1_input_acceleration
   - 0.307984579675928 x2_input_acceleration
   + 0.6901737526037663 x3_input_acceleration >= -0.6904024487016975
 CBF_Constraint_obs3: 0.2346984446428483 x1_input_acceleration
   - 0.3076059709723147 x2_input_acceleration
   + 0.6904839351751775 x3_input_acceleration >= -0.733091072538239
 CBF_Constraint_obs1: - 0.4653015553571517 x1_input_acceleration
   - 0.3076059709723147 x2_input_acceleration
   + 0.6904839351751775 x3_input_acceleration >= -0.9348687944132545
 CBF_Constraint_for_table: x3_input_acceleration >= -1.965400213742845
 Relaxed_CLF_constraint: - 0.0107543868399529 x1_input_acceleration
   + 0.00195788830839827 x2_input_acceleration
   + 0.3460054606486545 x3_input_acceleration + relaxation_CLF
   <= -20.04793570758465
 CBF_Constraint_obs2: - 0.1423535306373985 x1_input_acceleration
   - 0.3091651319555957 x2_input_acceleration
   + 0.6928183961823102 x3_input_acceleration >= -0.6948061707383526
 CBF_Constraint_obs3: 0.2376464693626015 x1_input_acceleration
   - 0.3091651319555957 x2_input_acceleration
   + 0.6928183961823102 x3_input_acceleration >= -0.7400703166328242
 CBF_Constraint_obs1: - 0.4623535306373985 x1_input_acceleration
   - 0.3091651319555957 x2_input_acceleration
   + 0.6928183961823102 x3_input_acceleration >= -0.9366889952482715
 CBF_Constraint_for_table: x3_input_acceleration >= -1.971641379821926
 Relaxed_CLF_constraint: - 0.00925438685495419 x1_input_acceleration
   + 0.0013676121685644 x2_input_acceleration
   + 0.3473277824379264 x3_input_acceleration + relaxation_CLF
   <= -20.04828964099694
 CBF_Constraint_obs2: - 0.1392128638554277 x1_input_acceleration
   - 0.3097267465496658 x2_input_acceleration
   + 0.6957659505674731 x3_input_acceleration >= -0.6992514212001627
 CBF_Constraint_obs3: 0.2406464691829023 x1_input_acceleration
   - 0.310135634412825 x2_input_acceleration
   + 0.6954630397627246 x3_input_acceleration >= -0.747204593173565
 CBF_Constraint_obs1: - 0.4593535308170977 x1_input_acceleration
   - 0.310135634412825 x2_input_acceleration
   + 0.6954630397627246 x3_input_acceleration >= -0.9385732721034858
 CBF_Constraint_for_table: x3_input_acceleration >= -1.978252988772962
 Relaxed_CLF_constraint: - 0.00775438694480379 x1_input_acceleration
   + 0.000882361 x2_input_acceleration
   + 0.3486501042281336 x3_input_acceleration + relaxation_CLF
   <= -20.0486471217024
 CBF_Constraint_obs2: - 0.1362128646020353 x1_input_acceleration
   - 0.3106539282630907 x2_input_acceleration
   + 0.6984105941499843 x3_input_acceleration >= -0.703546457061645
 CBF_Constraint_obs3: 0.2437871353979647 x1_input_acceleration
   - 0.3106539282630907 x2_input_acceleration
   + 0.6984105941499843 x3_input_acceleration >= -0.7546442356897114
 CBF_Constraint_obs1: - 0.4562128646020353 x1_input_acceleration
   - 0.3106539282630907 x2_input_acceleration
   + 0.6984105941499843 x3_input_acceleration >= -0.9405167487432731
 CBF_Constraint_for_table: x3_input_acceleration >= -1.985301201752219
 Relaxed_CLF_constraint: - 0.00625438731810757 x1_input_acceleration
   + 0.00041877 x2_input_acceleration
   + 0.3499724260193891 x3_input_acceleration + relaxation_CLF
   <= -20.0490079966812
 CBF_Constraint_obs2: - 0.1333234912561576 x1_input_acceleration
   - 0.3118826122864748 x2_input_acceleration
   + 0.7008186072892009 x3_input_acceleration >= -0.7077407968626768
 CBF_Constraint_obs3: 0.2467871349645455 x1_input_acceleration
   - 0.3115376206025149 x2_input_acceleration
   + 0.7010552377289592 x3_input_acceleration >= -0.761797531727366
 CBF_Constraint_obs1: - 0.4532128650354545 x1_input_acceleration
   - 0.3115376206025149 x2_input_acceleration
   + 0.7010552377289592 x3_input_acceleration >= -0.9424200455394114
 CBF_Constraint_for_table: x3_input_acceleration >= -1.991455643074229
 Relaxed_CLF_constraint: - 0.00475438753481716 x1_input_acceleration
   - 2.30761e-05 x2_input_acceleration
   + 0.3512947478088765 x3_input_acceleration + relaxation_CLF
   <= -20.04937224182857
 CBF_Constraint_obs2: - 0.130671191811022 x1_input_acceleration
   - 0.3126255309114063 x2_input_acceleration
   + 0.7034632508861689 x3_input_acceleration >= -0.7120877853678608
 CBF_Constraint_obs3: 0.249328808188978 x1_input_acceleration
   - 0.3126255309114063 x2_input_acceleration
   + 0.7034632508861689 x3_input_acceleration >= -0.7684501531473897
 CBF_Constraint_obs1: - 0.450671191811022 x1_input_acceleration
   - 0.3126255309114063 x2_input_acceleration
   + 0.7034632508861689 x3_input_acceleration >= -0.9446247388166784
 CBF_Constraint_for_table: x3_input_acceleration >= -1.998559586432494
 Relaxed_CLF_constraint: - 0.00342823781224935 x1_input_acceleration
   - 3.9453539894052092e-04 x2_input_acceleration
   + 0.3526170696073606 x3_input_acceleration + relaxation_CLF
   <= -20.04974028250047
 CBF_Constraint_obs2: - 0.1288637436445217 x1_input_acceleration
   - 0.3131688260982899 x2_input_acceleration
   + 0.7060242095433977 x3_input_acceleration >= -0.7164384531590519
 CBF_Constraint_obs3: 0.2511362563554783 x1_input_acceleration
   - 0.3131688260982899 x2_input_acceleration
   + 0.7060242095433977 x3_input_acceleration >= -0.7745178966967563
 CBF_Constraint_obs1: - 0.4488637436445217 x1_input_acceleration
   - 0.3131688260982899 x2_input_acceleration
   + 0.7060242095433977 x3_input_acceleration >= -0.9475294480746692
 CBF_Constraint_for_table: x3_input_acceleration >= -2.005171195378728
 Relaxed_CLF_constraint: - 0.00249542666234881 x1_input_acceleration
   - 6.3649987928415808e-04 x2_input_acceleration
   + 0.3539393913966074 x3_input_acceleration + relaxation_CLF
   <= -20.05011189002741
 CBF_Constraint_obs2: - 0.1275430721343231 x1_input_acceleration
   - 0.3134540638530317 x2_input_acceleration
   + 0.7086874497709372 x3_input_acceleration >= -0.7209482335728452
 CBF_Constraint_obs3: 0.2524569278656769 x1_input_acceleration
   - 0.3134540638530317 x2_input_acceleration
   + 0.7086874497709372 x3_input_acceleration >= -0.7802823150452383
 CBF_Constraint_obs1: - 0.4475430721343231 x1_input_acceleration
   - 0.3134540638530317 x2_input_acceleration
   + 0.7086874497709372 x3_input_acceleration >= -0.9509826912803038
 CBF_Constraint_for_table: x3_input_acceleration >= -2.011758366420294
 Relaxed_CLF_constraint: - 0.00183699825588232 x1_input_acceleration
   - 7.8460238397437543e-04 x2_input_acceleration
   + 0.3552617131857921 x3_input_acceleration + relaxation_CLF
   <= -20.05048595000768
 CBF_Constraint_obs2: - 0.1265041535096521 x1_input_acceleration
   - 0.3135862349180933 x2_input_acceleration
   + 0.7113766776235391 x3_input_acceleration >= -0.7254955063150048
 CBF_Constraint_obs3: 0.2533918959898596 x1_input_acceleration
   - 0.3136109112410304 x2_input_acceleration
   + 0.7113320933492471 x3_input_acceleration >= -0.785690754539234
 CBF_Constraint_obs1: - 0.4466081040101404 x1_input_acceleration
   - 0.3136109112410304 x2_input_acceleration
   + 0.7113320933492471 x3_input_acceleration >= -0.9547549365569796
 CBF_Constraint_for_table: x3_input_acceleration >= -2.017986419476939
 Relaxed_CLF_constraint: - 0.00136951419379097 x1_input_acceleration
   - 0.000863026 x2_input_acceleration
   + 0.3565840349749471 x3_input_acceleration + relaxation_CLF
   <= -20.05086191775286
 CBF_Constraint_obs2: - 0.1258340842988251 x1_input_acceleration
   - 0.3136368618577625 x2_input_acceleration
   + 0.7140213212054212 x3_input_acceleration >= -0.7300359312790554
 CBF_Constraint_obs3: 0.2541659157011749 x1_input_acceleration
   - 0.3136368618577625 x2_input_acceleration
   + 0.7140213212054212 x3_input_acceleration >= -0.7909935511951716
 CBF_Constraint_obs1: - 0.4458340842988251 x1_input_acceleration
   - 0.3136368618577625 x2_input_acceleration
   + 0.7140213212054212 x3_input_acceleration >= -0.9587031987181155
 CBF_Constraint_for_table: x3_input_acceleration >= -2.024580743084659
 Relaxed_CLF_constraint: - 0.00103447958837746 x1_input_acceleration
   - 8.8833954780831448e-04 x2_input_acceleration
   + 0.3579063567658881 x3_input_acceleration + relaxation_CLF
   <= -20.05123952780344
 CBF_Constraint_obs2: - 0.1253598143836702 x1_input_acceleration
   - 0.3132936244468882 x2_input_acceleration
   + 0.7169169003390563 x3_input_acceleration >= -0.7347972752164206
 CBF_Constraint_obs3: 0.2546401856163298 x1_input_acceleration
   - 0.3132936244468882 x2_input_acceleration
   + 0.7169169003390563 x3_input_acceleration >= -0.7962054515519339
 CBF_Constraint_obs1: - 0.4453598143836702 x1_input_acceleration
   - 0.3132936244468882 x2_input_acceleration
   + 0.7169169003390563 x3_input_acceleration >= -0.9630851267233567
 CBF_Constraint_for_table: x3_input_acceleration >= -2.031192352032486
 Relaxed_CLF_constraint: - 7.9090732916428652e-04 x1_input_acceleration
   - 8.7193133736629402e-04 x2_input_acceleration
   + 0.3592286785554534 x3_input_acceleration + relaxation_CLF
   <= -20.05161865171814
 CBF_Constraint_obs2: - 0.1249824347575268 x1_input_acceleration
   - 0.3131763326895198 x2_input_acceleration
   + 0.7195638088943561 x3_input_acceleration >= -0.7393401242816621
 CBF_Constraint_obs3: 0.2550175652424732 x1_input_acceleration
   - 0.3131763326895198 x2_input_acceleration
   + 0.7195638088943561 x3_input_acceleration >= -0.8011068112620117
 CBF_Constraint_obs1: - 0.4449824347575268 x1_input_acceleration
   - 0.3131763326895198 x2_input_acceleration
   + 0.7195638088943561 x3_input_acceleration >= -0.9673260720876836
 CBF_Constraint_for_table: x3_input_acceleration >= -2.037790549934408
 Relaxed_CLF_constraint: - 6.1008532920292513e-04 x1_input_acceleration
   - 8.2168971359541931e-04 x2_input_acceleration
   + 0.3605510003448527 x3_input_acceleration + relaxation_CLF
   <= -20.05199922849111
 CBF_Constraint_obs2: - 0.1247060605623977 x1_input_acceleration
   - 0.3130191345524123 x2_input_acceleration
   + 0.7222084524971328 x3_input_acceleration >= -0.7438970362440461
 CBF_Constraint_obs3: 0.2552939394376023 x1_input_acceleration
   - 0.3130191345524123 x2_input_acceleration
   + 0.7222084524971328 x3_input_acceleration >= -0.8059262787097683
 CBF_Constraint_obs1: - 0.4447060605623978 x1_input_acceleration
   - 0.3130191345524123 x2_input_acceleration
   + 0.7222084524971328 x3_input_acceleration >= -0.9716618846939642
 CBF_Constraint_for_table: x3_input_acceleration >= -2.044402158941349
 Relaxed_CLF_constraint: - 4.7189823163840128e-04 x1_input_acceleration
   - 7.4309064504163302e-04 x2_input_acceleration
   + 0.361873322146241 x3_input_acceleration + relaxation_CLF
   <= -20.05238123046112
 CBF_Constraint_obs2: - 0.1246107928883417 x1_input_acceleration
   - 0.3131975419648443 x2_input_acceleration
   + 0.7245044089153674 x3_input_acceleration >= -0.7481589856686093
 CBF_Constraint_obs3: 0.2553892071116583 x1_input_acceleration
   - 0.3131975419648443 x2_input_acceleration
   + 0.7245044089153674 x3_input_acceleration >= -0.8102787324246846
 CBF_Constraint_obs1: - 0.4446107928883417 x1_input_acceleration
   - 0.3131975419648443 x2_input_acceleration
   + 0.7245044089153674 x3_input_acceleration >= -0.9758476199792826
 CBF_Constraint_for_table: x3_input_acceleration >= -2.051431596476651
 Relaxed_CLF_constraint: - 3.6227556404397343e-04 x1_input_acceleration
   - 6.3989055792268434e-04 x2_input_acceleration
   + 0.3631956439413263 x3_input_acceleration + relaxation_CLF
   <= -20.05276464659259
 CBF_Constraint_obs2: - 0.1244241799488212 x1_input_acceleration
   - 0.3129218980135602 x2_input_acceleration
   + 0.727181954257575 x3_input_acceleration >= -0.7527438567627643
 CBF_Constraint_obs3: 0.2555758200511788 x1_input_acceleration
   - 0.3129218980135602 x2_input_acceleration
   + 0.727181954257575 x3_input_acceleration >= -0.8150408858113842
 CBF_Constraint_obs1: - 0.4444241799488212 x1_input_acceleration
   - 0.3129218980135602 x2_input_acceleration
   + 0.727181954257575 x3_input_acceleration >= -0.9802832007218212
 CBF_Constraint_for_table: x3_input_acceleration >= -2.057566964310625
 Relaxed_CLF_constraint: - 2.7135328007474313e-04 x1_input_acceleration
   - 5.1470476697301759e-04 x2_input_acceleration
   + 0.3645179657304721 x3_input_acceleration + relaxation_CLF
   <= -20.05314947435755
 CBF_Constraint_obs2: - 0.1242666359996832 x1_input_acceleration
   - 0.3126338884675346 x2_input_acceleration
   + 0.7298265978359055 x3_input_acceleration >= -0.7572862599371766
 CBF_Constraint_obs3: 0.2557333640003168 x1_input_acceleration
   - 0.3126338884675346 x2_input_acceleration
   + 0.7298265978359055 x3_input_acceleration >= -0.8197329557374775
 CBF_Constraint_obs1: - 0.4442666359996832 x1_input_acceleration
   - 0.3126338884675346 x2_input_acceleration
   + 0.7298265978359055 x3_input_acceleration >= -0.9846995687369231
 CBF_Constraint_for_table: x3_input_acceleration >= -2.064178573256451
 Relaxed_CLF_constraint: - 1.9258130550575681e-04 x1_input_acceleration
   - 0.0003707 x2_input_acceleration
   + 0.3658402875196374 x3_input_acceleration + relaxation_CLF
   <= -20.0535357161914
 CBF_Constraint_obs2: - 0.1240571354634012 x1_input_acceleration
   - 0.3120419342404908 x2_input_acceleration
   + 0.7327966827746044 x3_input_acceleration >= -0.7621891498367445
 CBF_Constraint_obs3: 0.2559428645365988 x1_input_acceleration
   - 0.3120419342404908 x2_input_acceleration
   + 0.7327966827746044 x3_input_acceleration >= -0.8248348711465134
 CBF_Constraint_obs1: - 0.444150595546409 x1_input_acceleration
   - 0.3124105293637818 x2_input_acceleration
   + 0.7324712414141308 x3_input_acceleration >= -0.9892302623504324
 CBF_Constraint_for_table: x3_input_acceleration >= -2.070831607430133
 Relaxed_CLF_constraint: - 0.000134561 x1_input_acceleration
   - 2.590204420838127e-04 x2_input_acceleration
   + 0.36716260930875 x3_input_acceleration + relaxation_CLF
   <= -20.05392338674907
 CBF_Constraint_obs2: - 0.1239898797951927 x1_input_acceleration
   - 0.3122199991743044 x2_input_acceleration
   + 0.7351641647546241 x3_input_acceleration >= -0.7666514591442414
 CBF_Constraint_obs3: 0.2560239484823952 x1_input_acceleration
   - 0.3118858555356936 x2_input_acceleration
   + 0.7354413263528297 x3_input_acceleration >= -0.8296187419841962
 CBF_Constraint_obs1: - 0.4439760515176048 x1_input_acceleration
   - 0.3118858555356936 x2_input_acceleration
   + 0.7354413263528297 x3_input_acceleration >= -0.9940768321400045
 CBF_Constraint_for_table: x3_input_acceleration >= -2.077443216375696
 Relaxed_CLF_constraint: - 9.40191e-05 x1_input_acceleration
   - 0.000180981 x2_input_acceleration
   + 0.3684849310978626 x3_input_acceleration + relaxation_CLF
   <= -20.05431247441598
 CBF_Constraint_obs2: - 0.1239332268685119 x1_input_acceleration
   - 0.3121109528204908 x2_input_acceleration
   + 0.7378088083328895 x3_input_acceleration >= -0.771418161557694
 CBF_Constraint_obs3: 0.2560667731314881 x1_input_acceleration
   - 0.3121109528204908 x2_input_acceleration
   + 0.7378088083328895 x3_input_acceleration >= -0.8341815960326077
 CBF_Constraint_obs1: - 0.4439193985909241 x1_input_acceleration
   - 0.31177680918188 x2_input_acceleration
   + 0.7380859699310951 x3_input_acceleration >= -0.9988001377476095
 CBF_Constraint_for_table: x3_input_acceleration >= -2.084061977878732
 Relaxed_CLF_constraint: - 6.56926e-05 x1_input_acceleration
   - 0.000126458 x2_input_acceleration
   + 0.3698072528869953 x3_input_acceleration + relaxation_CLF
   <= -20.05470296983798
 CBF_Constraint_obs2: - 0.1237954116716983 x1_input_acceleration
   - 0.3116737846967335 x2_input_acceleration
   + 0.7407391965781736 x3_input_acceleration >= -0.7764505117067864
 CBF_Constraint_obs3: 0.2562045883283017 x1_input_acceleration
   - 0.3116737846967335 x2_input_acceleration
   + 0.7407391965781736 x3_input_acceleration >= -0.8393448706186729
 CBF_Constraint_obs1: - 0.4437954116716983 x1_input_acceleration
   - 0.3116737846967335 x2_input_acceleration
   + 0.7407391965781736 x3_input_acceleration >= -1.003486841044145
 CBF_Constraint_for_table: x3_input_acceleration >= -2.090673586824309
 Relaxed_CLF_constraint: - 4.58993e-05 x1_input_acceleration
   - 8.83567e-05 x2_input_acceleration
   + 0.3711295746761107 x3_input_acceleration + relaxation_CLF
   <= -20.05509486844517
 CBF_Constraint_obs2: - 0.1237677522299372 x1_input_acceleration
   - 0.3116205421980032 x2_input_acceleration
   + 0.7433838401564119 x3_input_acceleration >= -0.7813066907719028
 CBF_Constraint_obs3: 0.2562322477700628 x1_input_acceleration
   - 0.3116205421980032 x2_input_acceleration
   + 0.7433838401564119 x3_input_acceleration >= -0.8442273261534625
 CBF_Constraint_obs1: - 0.4437677522299372 x1_input_acceleration
   - 0.3116205421980032 x2_input_acceleration
   + 0.7433838401564119 x3_input_acceleration >= -1.008320892555853
 CBF_Constraint_for_table: x3_input_acceleration >= -2.097285195769905
 Relaxed_CLF_constraint: - 3.20696e-05 x1_input_acceleration
   - 6.17355e-05 x2_input_acceleration
   + 0.3724518964652299 x3_input_acceleration + relaxation_CLF
   <= -20.05548816800811
 CBF_Constraint_obs2: - 0.1238991062880563 x1_input_acceleration
   - 0.3119971136764441 x2_input_acceleration
   + 0.7456930287938559 x3_input_acceleration >= -0.7859391008414568
 CBF_Constraint_obs3: 0.2561008937119437 x1_input_acceleration
   - 0.3119971136764441 x2_input_acceleration
   + 0.7456930287938559 x3_input_acceleration >= -0.8487349498678033
 CBF_Constraint_obs1: - 0.4438991062880563 x1_input_acceleration
   - 0.3119971136764441 x2_input_acceleration
   + 0.7456930287938559 x3_input_acceleration >= -1.013058385871902
 CBF_Constraint_for_table: x3_input_acceleration >= -2.103887565995565
 Relaxed_CLF_constraint: - 2.24063e-05 x1_input_acceleration
   - 4.31335e-05 x2_input_acceleration
   + 0.3737742182543499 x3_input_acceleration + relaxation_CLF
   <= -20.05588286743768
 CBF_Constraint_obs2: - 0.123827429328208 x1_input_acceleration
   - 0.3115896517182966 x2_input_acceleration
   + 0.7486321193172805 x3_input_acceleration >= -0.7910892417318056
 CBF_Constraint_obs3: 0.2561143965384912 x1_input_acceleration
   - 0.3119711214448591 x2_input_acceleration
   + 0.74833767237209 x3_input_acceleration >= -0.8536622957763951
 CBF_Constraint_obs1: - 0.4438856034615088 x1_input_acceleration
   - 0.3119711214448591 x2_input_acceleration
   + 0.74833767237209 x3_input_acceleration >= -1.017962101834035
 CBF_Constraint_for_table: x3_input_acceleration >= -2.110507519591419
 Relaxed_CLF_constraint: - 1.56549e-05 x1_input_acceleration
   - 3.01374e-05 x2_input_acceleration
   + 0.3750965400434669 x3_input_acceleration + relaxation_CLF
   <= -20.05627896620237
 CBF_Constraint_obs2: - 0.123788431684409 x1_input_acceleration
   - 0.3119467642308975 x2_input_acceleration
   + 0.7510528878497709 x3_input_acceleration >= -0.7958934998506164
 CBF_Constraint_obs3: 0.256211568315591 x1_input_acceleration
   - 0.3119467642308975 x2_input_acceleration
   + 0.7510528878497709 x3_input_acceleration >= -0.8587944897504279
 CBF_Constraint_obs1: - 0.443788431684409 x1_input_acceleration
   - 0.3119467642308975 x2_input_acceleration
   + 0.7510528878497709 x3_input_acceleration >= -1.022924245198144
 CBF_Constraint_for_table: x3_input_acceleration >= -2.117126281094459
 Relaxed_CLF_constraint: - 1.0938e-05 x1_input_acceleration
   - 2.10582e-05 x2_input_acceleration
   + 0.3764188618326003 x3_input_acceleration + relaxation_CLF
   <= -20.05667646404257
 CBF_Constraint_obs2: - 0.1237818403924241 x1_input_acceleration
   - 0.3119340768408037 x2_input_acceleration
   + 0.753697531428023 x3_input_acceleration >= -0.8008559764826945
 CBF_Constraint_obs3: 0.2562181596075759 x1_input_acceleration
   - 0.3119340768408037 x2_input_acceleration
   + 0.753697531428023 x3_input_acceleration >= -0.8637632281098916
 CBF_Constraint_obs1: - 0.4437818403924241 x1_input_acceleration
   - 0.3119340768408037 x2_input_acceleration
   + 0.753697531428023 x3_input_acceleration >= -1.027881448796634
 CBF_Constraint_for_table: x3_input_acceleration >= -2.123737890040089
 Relaxed_CLF_constraint: - 7.64238e-06 x1_input_acceleration
   - 1.47145e-05 x2_input_acceleration
   + 0.3777411836217264 x3_input_acceleration + relaxation_CLF
   <= -20.05707536083155
 CBF_Constraint_obs2: - 0.1238230210861028 x1_input_acceleration
   - 0.3118606404111979 x2_input_acceleration
   + 0.7563246512410682 x3_input_acceleration >= -0.8057702220793692
 CBF_Constraint_obs3: 0.2561769789138972 x1_input_acceleration
   - 0.3118606404111979 x2_input_acceleration
   + 0.7563246512410682 x3_input_acceleration >= -0.8686383520475714
 CBF_Constraint_obs1: - 0.4438230210861028 x1_input_acceleration
   - 0.3118606404111979 x2_input_acceleration
   + 0.7563246512410682 x3_input_acceleration >= -1.032828638948251
 CBF_Constraint_for_table: x3_input_acceleration >= -2.130810838937173
 Relaxed_CLF_constraint: - 5.34454e-06 x1_input_acceleration
   - 1.0302e-05 x2_input_acceleration
   + 0.3790635054110309 x3_input_acceleration + relaxation_CLF
   <= -20.05747565650768
 CBF_Constraint_obs2: - 0.1238198002370188 x1_input_acceleration
   - 0.3118544330578534 x2_input_acceleration
   + 0.7589692948193045 x3_input_acceleration >= -0.8107736510338535
 CBF_Constraint_obs3: 0.2561801997629812 x1_input_acceleration
   - 0.3118544330578534 x2_input_acceleration
   + 0.7589692948193045 x3_input_acceleration >= -0.8736448408086855
 CBF_Constraint_obs1: - 0.4438198002370188 x1_input_acceleration
   - 0.3118544330578534 x2_input_acceleration
   + 0.7589692948193045 x3_input_acceleration >= -1.037829491223468
 CBF_Constraint_for_table: x3_input_acceleration >= -2.137422447882765
 Relaxed_CLF_constraint: - 3.73412e-06 x1_input_acceleration
   - 7.19833e-06 x2_input_acceleration
   + 0.3803858272001491 x3_input_acceleration + relaxation_CLF
   <= -20.0578773510402
 CBF_Constraint_obs2: - 0.1237751112352191 x1_input_acceleration
   - 0.3118635658810509 x2_input_acceleration
   + 0.7616240711871431 x3_input_acceleration >= -0.8158129846212674
 CBF_Constraint_obs3: 0.2562248887647809 x1_input_acceleration
   - 0.3118635658810509 x2_input_acceleration
   + 0.7616240711871431 x3_input_acceleration >= -0.8787266289478092
 CBF_Constraint_obs1: - 0.4437751112352191 x1_input_acceleration
   - 0.3118635658810509 x2_input_acceleration
   + 0.7616240711871431 x3_input_acceleration >= -1.042833073609443
 CBF_Constraint_for_table: x3_input_acceleration >= -2.143615632222009
 Relaxed_CLF_constraint: - 2.60887e-06 x1_input_acceleration
   - 5.02942e-06 x2_input_acceleration
   + 0.3817081489892625 x3_input_acceleration + relaxation_CLF
   <= -20.05828044441476
 CBF_Constraint_obs2: - 0.1237077355362931 x1_input_acceleration
   - 0.3115063650269081 x2_input_acceleration
   + 0.7645423000848923 x3_input_acceleration >= -0.8210809348786153
 CBF_Constraint_obs3: 0.2562922644637069 x1_input_acceleration
   - 0.3115063650269081 x2_input_acceleration
   + 0.7645423000848923 x3_input_acceleration >= -0.8840585861191369
 CBF_Constraint_obs1: - 0.4437077355362931 x1_input_acceleration
   - 0.3115063650269081 x2_input_acceleration
   + 0.7645423000848923 x3_input_acceleration >= -1.04804712330765
 CBF_Constraint_for_table: x3_input_acceleration >= -2.150551192411938
 Relaxed_CLF_constraint: - 1.82279e-06 x1_input_acceleration
   - 3.51439e-06 x2_input_acceleration
   + 0.3830304707783776 x3_input_acceleration + relaxation_CLF
   <= -20.05868493662415
 CBF_Constraint_obs2: - 0.1237066636401368 x1_input_acceleration
   - 0.311504357999054 x2_input_acceleration
   + 0.7671869436642962 x3_input_acceleration >= -0.826142637762926
 CBF_Constraint_obs3: 0.2562933363598632 x1_input_acceleration
   - 0.311504357999054 x2_input_acceleration
   + 0.7671869436642962 x3_input_acceleration >= -0.889121307304796
 CBF_Constraint_obs1: - 0.4437066636401368 x1_input_acceleration
   - 0.311504357999054 x2_input_acceleration
   + 0.7671869436642962 x3_input_acceleration >= -1.053107968675035
 CBF_Constraint_for_table: x3_input_acceleration >= -2.157162801360448
 Relaxed_CLF_constraint: - 1.28684e-06 x1_input_acceleration
   - 2.51088e-06 x2_input_acceleration
   + 0.3843527925680795 x3_input_acceleration + relaxation_CLF
   <= -20.05909082766514
 CBF_Constraint_obs2: - 0.1238565870351266 x1_input_acceleration
   - 0.3119156262094429 x2_input_acceleration
   + 0.7694944633725034 x3_input_acceleration >= -0.8309419264831946
 CBF_Constraint_obs3: 0.2561434129648734 x1_input_acceleration
   - 0.3119156262094429 x2_input_acceleration
   + 0.7694944633725034 x3_input_acceleration >= -0.8937781687998245
 CBF_Constraint_obs1: - 0.4438565870351266 x1_input_acceleration
   - 0.3119156262094429 x2_input_acceleration
   + 0.7694944633725034 x3_input_acceleration >= -1.058027196111296
 CBF_Constraint_for_table: x3_input_acceleration >= -2.163810769141403
 Relaxed_CLF_constraint: - 9.08264e-07 x1_input_acceleration
   - 1.7937e-06 x2_input_acceleration
   + 0.3856751143576079 x3_input_acceleration + relaxation_CLF
   <= -20.05949811753552
 CBF_Constraint_obs2: - 0.1238560405201539 x1_input_acceleration
   - 0.3119145492329467 x2_input_acceleration
   + 0.772139106950777 x3_input_acceleration >= -0.8360372565990823
 CBF_Constraint_obs3: 0.2561439594798461 x1_input_acceleration
   - 0.3119145492329467 x2_input_acceleration
   + 0.772139106950777 x3_input_acceleration >= -0.8988740181049361
 CBF_Constraint_obs1: - 0.4438560405201539 x1_input_acceleration
   - 0.3119145492329467 x2_input_acceleration
   + 0.772139106950777 x3_input_acceleration >= -1.063122089015205
 CBF_Constraint_for_table: x3_input_acceleration >= -2.170422378087087
 Relaxed_CLF_constraint: - 6.35006e-07 x1_input_acceleration
   - 1.25521e-06 x2_input_acceleration
   + 0.3869974361467448 x3_input_acceleration + relaxation_CLF
   <= -20.05990680623445
 CBF_Constraint_obs2: - 0.1238761687333024 x1_input_acceleration
   - 0.3119011870434611 x2_input_acceleration
   + 0.7747813663437808 x3_input_acceleration >= -0.841142276615872
 CBF_Constraint_obs3: 0.2561238312666976 x1_input_acceleration
   - 0.3119011870434611 x2_input_acceleration
   + 0.7747813663437808 x3_input_acceleration >= -0.9039599163192347
 CBF_Constraint_obs1: - 0.4438761687333024 x1_input_acceleration
   - 0.3119011870434611 x2_input_acceleration
   + 0.7747813663437808 x3_input_acceleration >= -1.068243211602514
 CBF_Constraint_for_table: x3_input_acceleration >= -2.177043821800461
 Relaxed_CLF_constraint: - 4.47114e-07 x1_input_acceleration
   - 8.92213e-07 x2_input_acceleration
   + 0.3883197579361421 x3_input_acceleration + relaxation_CLF
   <= -20.06031689376183
 CBF_Constraint_obs2: - 0.1237819857106949 x1_input_acceleration
   - 0.311536326557744 x2_input_acceleration
   + 0.7777601535619085 x3_input_acceleration >= -0.8466096490252744
 CBF_Constraint_obs3: 0.2561240773691391 x1_input_acceleration
   - 0.3119007493559007 x2_input_acceleration
   + 0.7774260099232977 x3_input_acceleration >= -0.9090910267027805
 CBF_Constraint_obs1: - 0.443875922630861 x1_input_acceleration
   - 0.3119007493559007 x2_input_acceleration
   + 0.7774260099232977 x3_input_acceleration >= -1.073373891306787
 CBF_Constraint_for_table: x3_input_acceleration >= -2.183719803765611
 Relaxed_CLF_constraint: - 3.24063e-07 x1_input_acceleration
   - 6.73369e-07 x2_input_acceleration
   + 0.3896420797259006 x3_input_acceleration + relaxation_CLF
   <= -20.06072838011747
 CBF_Constraint_obs2: - 0.1237817907231578 x1_input_acceleration
   - 0.3115359222499163 x2_input_acceleration
   + 0.7804047971401471 x3_input_acceleration >= -0.8517602624576146
 CBF_Constraint_obs3: 0.2562182092768422 x1_input_acceleration
   - 0.3115359222499163 x2_input_acceleration
   + 0.7804047971401471 x3_input_acceleration >= -0.9146675612706145
 CBF_Constraint_obs1: - 0.4437817907231578 x1_input_acceleration
   - 0.3115359222499163 x2_input_acceleration
   + 0.7804047971401471 x3_input_acceleration >= -1.078785695036141
 CBF_Constraint_for_table: x3_input_acceleration >= -2.190331412711208
 Relaxed_CLF_constraint: - 2.26569e-07 x1_input_acceleration
   - 4.71215e-07 x2_input_acceleration
   + 0.3909644015150199 x3_input_acceleration + relaxation_CLF
   <= -20.06114126530091
 CBF_Constraint_obs2: - 0.1238722689429929 x1_input_acceleration
   - 0.3119077580365124 x2_input_acceleration
   + 0.7826847795026908 x3_input_acceleration >= -0.8565328157520304
 CBF_Constraint_obs3: 0.2561277310570071 x1_input_acceleration
   - 0.3119077580365124 x2_input_acceleration
   + 0.7826847795026908 x3_input_acceleration >= -0.9193541602561872
 CBF_Constraint_obs1: - 0.4438722689429929 x1_input_acceleration
   - 0.3119077580365124 x2_input_acceleration
   + 0.7826847795026908 x3_input_acceleration >= -1.083630630906425
 CBF_Constraint_for_table: x3_input_acceleration >= -2.196414924506697
 Relaxed_CLF_constraint: - 1.66149e-07 x1_input_acceleration
   - 3.63012e-07 x2_input_acceleration
   + 0.3922867233046596 x3_input_acceleration + relaxation_CLF
   <= -20.06155554931251
 CBF_Constraint_obs2: - 0.1238721705690841 x1_input_acceleration
   - 0.3119075471861388 x2_input_acceleration
   + 0.7853294230810628 x3_input_acceleration >= -0.8617161692375057
 CBF_Constraint_obs3: 0.2561278294309159 x1_input_acceleration
   - 0.3119075471861388 x2_input_acceleration
   + 0.7853294230810628 x3_input_acceleration >= -0.9245376071968758
 CBF_Constraint_obs1: - 0.4438721705690841 x1_input_acceleration
   - 0.3119075471861388 x2_input_acceleration
   + 0.7853294230810628 x3_input_acceleration >= -1.088813905692773
 CBF_Constraint_for_table: x3_input_acceleration >= -2.203026533452626
 Relaxed_CLF_constraint: - 1.16962e-07 x1_input_acceleration
   - 2.57587e-07 x2_input_acceleration
   + 0.3936090450938456 x3_input_acceleration + relaxation_CLF
   <= -20.06197123215191
 CBF_Constraint_obs2: - 0.1238721115694994 x1_input_acceleration
   - 0.3119074416567453 x2_input_acceleration
   + 0.7879740666601978 x3_input_acceleration >= -0.8669171023931903
 CBF_Constraint_obs3: 0.2561278884305006 x1_input_acceleration
   - 0.3119074416567453 x2_input_acceleration
   + 0.7879740666601978 x3_input_acceleration >= -0.9297385964021658
 CBF_Constraint_obs1: - 0.4438721115694994 x1_input_acceleration
   - 0.3119074416567453 x2_input_acceleration
   + 0.7879740666601978 x3_input_acceleration >= -1.09401479164879
 CBF_Constraint_for_table: x3_input_acceleration >= -2.209618770890912
 Relaxed_CLF_constraint: - 8.7462e-08 x1_input_acceleration
   - 2.04822e-07 x2_input_acceleration
   + 0.3949313668834131 x3_input_acceleration + relaxation_CLF
   <= -20.06238831381938
 CBF_Constraint_obs2: - 0.1238429775946389 x1_input_acceleration
   - 0.3119248677285646 x2_input_acceleration
   + 0.7906398102834085 x3_input_acceleration >= -0.8721818947649662
 CBF_Constraint_obs3: 0.2561570224053611 x1_input_acceleration
   - 0.3119248677285646 x2_input_acceleration
   + 0.7906398102834085 x3_input_acceleration >= -0.9350310660500593
 CBF_Constraint_obs1: - 0.4438429775946389 x1_input_acceleration
   - 0.3119248677285646 x2_input_acceleration
   + 0.7906398102834085 x3_input_acceleration >= -1.099256276840677
 CBF_Constraint_for_table: x3_input_acceleration >= -2.216300117272699
 Relaxed_CLF_constraint: - 6.40079e-08 x1_input_acceleration
   - 1.55975e-07 x2_input_acceleration
   + 0.3962536886728932 x3_input_acceleration + relaxation_CLF
   <= -20.06280679431476
 CBF_Constraint_obs2: - 0.1233394016491191 x1_input_acceleration
   - 0.3114213647457553 x2_input_acceleration
   + 0.7937679667403337 x3_input_acceleration >= -0.8778293243029436
 CBF_Constraint_obs3: 0.2561570583118185 x1_input_acceleration
   - 0.3119247855755283 x2_input_acceleration
   + 0.7932844538619157 x3_input_acceleration >= -0.9402671688978769
 CBF_Constraint_obs1: - 0.4438429416881816 x1_input_acceleration
   - 0.3119247855755283 x2_input_acceleration
   + 0.7932844538619157 x3_input_acceleration >= -1.104492316852195
 CBF_Constraint_for_table: x3_input_acceleration >= -2.222911726218968
 Relaxed_CLF_constraint: - 4.60547e-08 x1_input_acceleration
   - 1.14899e-07 x2_input_acceleration
   + 0.3975760104621469 x3_input_acceleration + relaxation_CLF
   <= -20.06322667363801
 CBF_Constraint_obs2: - 0.123339378754961 x1_input_acceleration
   - 0.3114213167123168 x2_input_acceleration
   + 0.7964126103190237 x3_input_acceleration >= -0.8830861059120672
 CBF_Constraint_obs3: 0.256660621245039 x1_input_acceleration
   - 0.3114213167123168 x2_input_acceleration
   + 0.7964126103190237 x3_input_acceleration >= -0.9464136960948543
 CBF_Constraint_obs1: - 0.443339378754961 x1_input_acceleration
   - 0.3114213167123168 x2_input_acceleration
   + 0.7964126103190237 x3_input_acceleration >= -1.109757608916036
 CBF_Constraint_for_table: x3_input_acceleration >= -2.229531083769514
 Relaxed_CLF_constraint: - 3.46076e-08 x1_input_acceleration
   - 9.08818e-08 x2_input_acceleration
   + 0.3988983322514918 x3_input_acceleration + relaxation_CLF
   <= -20.06364795178921
 CBF_Constraint_obs2: - 0.1233393637641428 x1_input_acceleration
   - 0.3114212877705496 x2_input_acceleration
   + 0.7990572538979359 x3_input_acceleration >= -0.8883603901716517
 CBF_Constraint_obs3: 0.2566606362358572 x1_input_acceleration
   - 0.3114212877705496 x2_input_acceleration
   + 0.7990572538979359 x3_input_acceleration >= -0.951687994595716
 CBF_Constraint_obs1: - 0.4433393637641428 x1_input_acceleration
   - 0.3114212877705496 x2_input_acceleration
   + 0.7990572538979359 x3_input_acceleration >= -1.115031881182966
 CBF_Constraint_for_table: x3_input_acceleration >= -2.23610543981381
 Relaxed_CLF_constraint: - 2.71122e-08 x1_input_acceleration
   - 7.64109e-08 x2_input_acceleration
   + 0.4002206540409479 x3_input_acceleration + relaxation_CLF
   <= -20.06407062876839
 CBF_Constraint_obs2: - 0.1238662530397379 x1_input_acceleration
   - 0.3118766236879409 x2_input_acceleration
   + 0.8012555778961234 x3_input_acceleration >= -0.8932754726934307
 CBF_Constraint_obs3: 0.2561337469602621 x1_input_acceleration
   - 0.3118766236879409 x2_input_acceleration
   + 0.8012555778961234 x3_input_acceleration >= -0.9561025323056798
 CBF_Constraint_obs1: - 0.4438662530397379 x1_input_acceleration
   - 0.3118766236879409 x2_input_acceleration
   + 0.8012555778961234 x3_input_acceleration >= -1.120368475125221
 CBF_Constraint_for_table: x3_input_acceleration >= -2.242727777595533
 Relaxed_CLF_constraint: - 1.92201e-08 x1_input_acceleration
   - 5.46266e-08 x2_input_acceleration
   + 0.4015429758300808 x3_input_acceleration + relaxation_CLF
   <= -20.06449470457539
 CBF_Constraint_obs2: - 0.1237804108017028 x1_input_acceleration
   - 0.3118609745024286 x2_input_acceleration
   + 0.8039374147726863 x3_input_acceleration >= -0.8986177804840412
 CBF_Constraint_obs3: 0.2562195891982972 x1_input_acceleration
   - 0.3118609745024286 x2_input_acceleration
   + 0.8039374147726863 x3_input_acceleration >= -0.9615263902224235
 CBF_Constraint_obs1: - 0.4437804108017028 x1_input_acceleration
   - 0.3118609745024286 x2_input_acceleration
   + 0.8039374147726863 x3_input_acceleration >= -1.125642109125403
 CBF_Constraint_for_table: x3_input_acceleration >= -2.249339386541091
 Relaxed_CLF_constraint: - 1.34453e-08 x1_input_acceleration
   - 3.82423e-08 x2_input_acceleration
   + 0.4028652976191923 x3_input_acceleration + relaxation_CLF
   <= -20.06492017921032
 CBF_Constraint_obs2: - 0.1233355196923087 x1_input_acceleration
   - 0.3114532673664165 x2_input_acceleration
   + 0.8070109733789865 x3_input_acceleration >= -0.9043518741563595
 CBF_Constraint_obs3: 0.2566644803076913 x1_input_acceleration
   - 0.3114532673664165 x2_input_acceleration
   + 0.8070109733789865 x3_input_acceleration >= -0.9676831304486664
 CBF_Constraint_obs1: - 0.4433355196923087 x1_input_acceleration
   - 0.3114532673664165 x2_input_acceleration
   + 0.8070109733789865 x3_input_acceleration >= -1.131020289910206
 CBF_Constraint_for_table: x3_input_acceleration >= -2.25633902173482
 Relaxed_CLF_constraint: - 1.24249e-08 x1_input_acceleration
   - 3.25594e-08 x2_input_acceleration
   + 0.4041876194104406 x3_input_acceleration + relaxation_CLF
   <= -20.06534705267387
 CBF_Constraint_obs2: - 0.1233355200876457 x1_input_acceleration
   - 0.3114532494351967 x2_input_acceleration
   + 0.8096556169572406 x3_input_acceleration >= -0.9096962439618037
 CBF_Constraint_obs3: 0.2566644799123543 x1_input_acceleration
   - 0.3114532494351967 x2_input_acceleration
   + 0.8096556169572406 x3_input_acceleration >= -0.9730274998785402
 CBF_Constraint_obs1: - 0.4433355200876457 x1_input_acceleration
   - 0.3114532494351967 x2_input_acceleration
   + 0.8096556169572406 x3_input_acceleration >= -1.13636466003192
 CBF_Constraint_for_table: x3_input_acceleration >= -2.262950630680455
 Relaxed_CLF_constraint: - 1.26226e-08 x1_input_acceleration
   - 2.35938e-08 x2_input_acceleration
   + 0.4055099411995676 x3_input_acceleration + relaxation_CLF
   <= -20.06577532496467
 CBF_Constraint_obs2: - 0.1236631039810339 x1_input_acceleration
   - 0.3115510059818209 x2_input_acceleration
   + 0.81219738291913 x3_input_acceleration >= -0.9150264767939906
 CBF_Constraint_obs3: 0.2563368960189661 x1_input_acceleration
   - 0.3115510059818209 x2_input_acceleration
   + 0.81219738291913 x3_input_acceleration >= -0.9780465280120085
 CBF_Constraint_obs1: - 0.4436631039810339 x1_input_acceleration
   - 0.3115510059818209 x2_input_acceleration
   + 0.81219738291913 x3_input_acceleration >= -1.141956959978818
 CBF_Constraint_for_table: x3_input_acceleration >= -2.269168550948643
 Relaxed_CLF_constraint: - 1.10054e-08 x1_input_acceleration
   - 2.60584e-08 x2_input_acceleration
   + 0.4068322629889534 x3_input_acceleration + relaxation_CLF
   <= -20.06620499608348
 CBF_Constraint_obs2: - 0.1238662390114403 x1_input_acceleration
   - 0.3118834755009172 x2_input_acceleration
   + 0.8144853523032731 x3_input_acceleration >= -0.9200006707174468
 CBF_Constraint_obs3: 0.2563368936179543 x1_input_acceleration
   - 0.3115510007923601 x2_input_acceleration
   + 0.8148420264976091 x3_input_acceleration >= -0.9834251965889422
 CBF_Constraint_obs1: - 0.4436631063820458 x1_input_acceleration
   - 0.3115510007923601 x2_input_acceleration
   + 0.8148420264976091 x3_input_acceleration >= -1.147335632757522
 CBF_Constraint_for_table: x3_input_acceleration >= -2.275729197923558
 Relaxed_CLF_constraint: - 1.22059e-08 x1_input_acceleration
   - 2.34637e-08 x2_input_acceleration
   + 0.408154584778193 x3_input_acceleration + relaxation_CLF
   <= -20.06663606603018
 CBF_Constraint_obs2: - 0.1238662399298998 x1_input_acceleration
   - 0.3118834653144602 x2_input_acceleration
   + 0.8171299958815843 x3_input_acceleration >= -0.9253944643754569
 CBF_Constraint_obs3: 0.2561337600701002 x1_input_acceleration
   - 0.3118834653144602 x2_input_acceleration
   + 0.8171299958815843 x3_input_acceleration >= -0.9882215364420521
 CBF_Constraint_obs1: - 0.4438662399298998 x1_input_acceleration
   - 0.3118834653144602 x2_input_acceleration
   + 0.8171299958815843 x3_input_acceleration >= -1.152487456319377
 CBF_Constraint_for_table: x3_input_acceleration >= -2.282340806869335
 Relaxed_CLF_constraint: - 1.26651e-08 x1_input_acceleration
   - 1.83704e-08 x2_input_acceleration
   + 0.4094769065673485 x3_input_acceleration + relaxation_CLF
   <= -20.06706853480479
 CBF_Constraint_obs2: - 0.1237685013219724 x1_input_acceleration
   - 0.3119251455567564 x2_input_acceleration
   + 0.8197858451362294 x3_input_acceleration >= -0.9308309627947658
 CBF_Constraint_obs3: 0.2562314986780276 x1_input_acceleration
   - 0.3119251455567564 x2_input_acceleration
   + 0.8197858451362294 x3_input_acceleration >= -0.9937508865388921
 CBF_Constraint_obs1: - 0.443866252939404 x1_input_acceleration
   - 0.3118835415147032 x2_input_acceleration
   + 0.8197746394630117 x3_input_acceleration >= -1.157898816841164
 CBF_Constraint_for_table: x3_input_acceleration >= -2.28905582988159
 Relaxed_CLF_constraint: - 1.91699e-08 x1_input_acceleration
   - 5.64705e-08 x2_input_acceleration
   + 0.4107992283580623 x3_input_acceleration + relaxation_CLF
   <= -20.06750240240783
 CBF_Constraint_obs2: - 0.1237684903111004 x1_input_acceleration
   - 0.3119251138714427 x2_input_acceleration
   + 0.8224304887144998 x3_input_acceleration >= -0.9362597807809654
 CBF_Constraint_obs3: 0.2562315096888996 x1_input_acceleration
   - 0.3119251138714427 x2_input_acceleration
   + 0.8224304887144998 x3_input_acceleration >= -0.99917971498542
 CBF_Constraint_obs1: - 0.4437684903111004 x1_input_acceleration
   - 0.3119251138714427 x2_input_acceleration
   + 0.8224304887144998 x3_input_acceleration >= -1.163274573029846
 CBF_Constraint_for_table: x3_input_acceleration >= -2.295667438827265
 Relaxed_CLF_constraint: - 1.36645e-08 x1_input_acceleration
   - 4.06279e-08 x2_input_acceleration
   + 0.4121215501471974 x3_input_acceleration + relaxation_CLF
   <= -20.06793766883829
 CBF_Constraint_obs2: - 0.1238547935263678 x1_input_acceleration
   - 0.3118889859030529 x2_input_acceleration
   + 0.8250639266201452 x3_input_acceleration >= -0.9416815405211827
 CBF_Constraint_obs3: 0.256231513999267 x1_input_acceleration
   - 0.3119251063177868 x2_input_acceleration
   + 0.8250751322933629 x3_input_acceleration >= -1.004626043308287
 CBF_Constraint_obs1: - 0.443768486000733 x1_input_acceleration
   - 0.3119251063177868 x2_input_acceleration
   + 0.8250751322933629 x3_input_acceleration >= -1.16872089380957
 CBF_Constraint_for_table: x3_input_acceleration >= -2.302250139521707
 Relaxed_CLF_constraint: - 1.15093e-08 x1_input_acceleration
   - 3.68511e-08 x2_input_acceleration
   + 0.413443871936629 x3_input_acceleration + relaxation_CLF
   <= -20.06837433409678
 CBF_Constraint_obs2: - 0.1238600388808085 x1_input_acceleration
   - 0.3118858645242666 x2_input_acceleration
   + 0.8276858012241697 x3_input_acceleration >= -0.9470973590871254
 CBF_Constraint_obs3: 0.2561452063279317 x1_input_acceleration
   - 0.3118889639657949 x2_input_acceleration
   + 0.8277085701984739 x3_input_acceleration >= -1.009983212185547
 CBF_Constraint_obs1: - 0.4438547936720683 x1_input_acceleration
   - 0.3118889639657949 x2_input_acceleration
   + 0.8277085701984739 x3_input_acceleration >= -1.174229101111667
 CBF_Constraint_for_table: x3_input_acceleration >= -2.308837608586395
 Relaxed_CLF_constraint: - 1.15821e-08 x1_input_acceleration
   - 2.58824e-08 x2_input_acceleration
   + 0.4147661937257933 x3_input_acceleration + relaxation_CLF
   <= -20.06881239818311
 CBF_Constraint_obs2: - 0.1238600427338598 x1_input_acceleration
   - 0.3118858655885026 x2_input_acceleration
   + 0.8303304448031252 x3_input_acceleration >= -0.9525784386335985
 CBF_Constraint_obs3: 0.2561399572661402 x1_input_acceleration
   - 0.3118858655885026 x2_input_acceleration
   + 0.8303304448031252 x3_input_acceleration >= -1.015411398036432
 CBF_Constraint_obs1: - 0.4438600427338598 x1_input_acceleration
   - 0.3118858655885026 x2_input_acceleration
   + 0.8303304448031252 x3_input_acceleration >= -1.179666472820686
 CBF_Constraint_for_table: x3_input_acceleration >= -2.315449217533784
 Relaxed_CLF_constraint: - 1.35087e-08 x1_input_acceleration
   - 2.64146e-08 x2_input_acceleration
   + 0.4160885155152712 x3_input_acceleration + relaxation_CLF
   <= -20.06925186109748
 CBF_Constraint_obs2: - 0.12378040991478 x1_input_acceleration
   - 0.311536001233029 x2_input_acceleration
   + 0.833288012767762 x3_input_acceleration >= -0.9584314777067283
 CBF_Constraint_obs3: 0.25621959008522 x1_input_acceleration
   - 0.311536001233029 x2_input_acceleration
   + 0.833288012767762 x3_input_acceleration >= -1.021340088287688
 CBF_Constraint_obs1: - 0.44378040991478 x1_input_acceleration
   - 0.311536001233029 x2_input_acceleration
   + 0.833288012767762 x3_input_acceleration >= -1.185455805638552
 CBF_Constraint_for_table: x3_input_acceleration >= -2.322074237527773
 Relaxed_CLF_constraint: - 1.30018e-08 x1_input_acceleration
   - 3.38692e-08 x2_input_acceleration
   + 0.4174108373050541 x3_input_acceleration + relaxation_CLF
   <= -20.06969272283988
 CBF_Constraint_obs2: - 0.1237804030576438 x1_input_acceleration
   - 0.3115359850521529 x2_input_acceleration
   + 0.8359326563461063 x3_input_acceleration >= -0.9639495801367316
 CBF_Constraint_obs3: 0.2562195969423562 x1_input_acceleration
   - 0.3115359850521529 x2_input_acceleration
   + 0.8359326563461063 x3_input_acceleration >= -1.02685819723197
 CBF_Constraint_obs1: - 0.4437804030576438 x1_input_acceleration
   - 0.3115359850521529 x2_input_acceleration
   + 0.8359326563461063 x3_input_acceleration >= -1.190973902582847
 CBF_Constraint_for_table: x3_input_acceleration >= -2.328685846473634
 Relaxed_CLF_constraint: - 9.57326e-09 x1_input_acceleration
   - 2.57788e-08 x2_input_acceleration
   + 0.4187331590942263 x3_input_acceleration + relaxation_CLF
   <= -20.07013498341001
 CBF_Constraint_obs2: - 0.1237088776949075 x1_input_acceleration
   - 0.3114977042438019 x2_input_acceleration
   + 0.8385823067145102 x3_input_acceleration >= -0.9694437391304018
 CBF_Constraint_obs3: 0.256219596731362 x1_input_acceleration
   - 0.3115359704257477 x2_input_acceleration
   + 0.838577299924349 x3_input_acceleration >= -1.032393788208382
 CBF_Constraint_obs1: - 0.443780403268638 x1_input_acceleration
   - 0.3115359704257477 x2_input_acceleration
   + 0.838577299924349 x3_input_acceleration >= -1.196509493928499
 CBF_Constraint_for_table: x3_input_acceleration >= -2.335641374219594
 Relaxed_CLF_constraint: - 9.67875e-09 x1_input_acceleration
   - 1.84656e-08 x2_input_acceleration
   + 0.4200554808833476 x3_input_acceleration + relaxation_CLF
   <= -20.07057864280806
 CBF_Constraint_obs2: - 0.1238385820330586 x1_input_acceleration
   - 0.3118642813130268 x2_input_acceleration
   + 0.8409551531134691 x3_input_acceleration >= -0.9747511173836845
 CBF_Constraint_obs3: 0.2561614179669414 x1_input_acceleration
   - 0.3118642813130268 x2_input_acceleration
   + 0.8409551531134691 x3_input_acceleration >= -1.037604464452279
 CBF_Constraint_obs1: - 0.4438385820330586 x1_input_acceleration
   - 0.3118642813130268 x2_input_acceleration
   + 0.8409551531134691 x3_input_acceleration >= -1.201821983010132
 CBF_Constraint_for_table: x3_input_acceleration >= -2.342306627347729
 Relaxed_CLF_constraint: - 1.19943e-08 x1_input_acceleration
   - 2.27175e-08 x2_input_acceleration
   + 0.421377802672915 x3_input_acceleration + relaxation_CLF
   <= -20.07102370103418
 CBF_Constraint_obs2: - 0.1238385823446349 x1_input_acceleration
   - 0.3118642689618898 x2_input_acceleration
   + 0.8435997966917261 x3_input_acceleration >= -0.9803199171382305
 CBF_Constraint_obs3: 0.2561614176553652 x1_input_acceleration
   - 0.3118642689618898 x2_input_acceleration
   + 0.8435997966917261 x3_input_acceleration >= -1.043173263910827
 CBF_Constraint_obs1: - 0.4438385823446349 x1_input_acceleration
   - 0.3118642689618898 x2_input_acceleration
   + 0.8435997966917261 x3_input_acceleration >= -1.207390783013939
 CBF_Constraint_for_table: x3_input_acceleration >= -2.348918236293372
 Relaxed_CLF_constraint: - 1.21501e-08 x1_input_acceleration
   - 1.65419e-08 x2_input_acceleration
   + 0.4227001244620435 x3_input_acceleration + relaxation_CLF
   <= -20.07147015808809
 CBF_Constraint_obs2: - 0.1238691018986831 x1_input_acceleration
   - 0.311891803811019 x2_input_acceleration
   + 0.846207366183042 x3_input_acceleration >= -0.985858697840148
 CBF_Constraint_obs3: 0.2561308981013169 x1_input_acceleration
   - 0.311891803811019 x2_input_acceleration
   + 0.846207366183042 x3_input_acceleration >= -1.048683051036399
 CBF_Constraint_obs1: - 0.4438385843205581 x1_input_acceleration
   - 0.3118642664651328 x2_input_acceleration
   + 0.8462444402720923 x3_input_acceleration >= -1.212977077901174
 CBF_Constraint_for_table: x3_input_acceleration >= -2.355493486410974
 Relaxed_CLF_constraint: - 1.31381e-08 x1_input_acceleration
   - 1.52936e-08 x2_input_acceleration
   + 0.4240224462522266 x3_input_acceleration + relaxation_CLF
   <= -20.07191801397029
 CBF_Constraint_obs2: - 0.1238690940734471 x1_input_acceleration
   - 0.3118917950020412 x2_input_acceleration
   + 0.8488520097612722 x3_input_acceleration >= -0.9914622234149033
 CBF_Constraint_obs3: 0.2561309059265529 x1_input_acceleration
   - 0.3118917950020412 x2_input_acceleration
   + 0.8488520097612722 x3_input_acceleration >= -1.054286584045129
 CBF_Constraint_obs1: - 0.4438690940734471 x1_input_acceleration
   - 0.3118917950020412 x2_input_acceleration
   + 0.8488520097612722 x3_input_acceleration >= -1.218557498673661
 CBF_Constraint_for_table: x3_input_acceleration >= -2.362105095356549
 Relaxed_CLF_constraint: - 9.22544e-09 x1_input_acceleration
   - 1.08891e-08 x2_input_acceleration
   + 0.4253447680413416 x3_input_acceleration + relaxation_CLF
   <= -20.07236726868006
 CBF_Constraint_obs2: - 0.1238691038912369 x1_input_acceleration
   - 0.3118918359234391 x2_input_acceleration
   + 0.8514966533422488 x3_input_acceleration >= -0.9970832785844514
 CBF_Constraint_obs3: 0.2561308961087631 x1_input_acceleration
   - 0.3118918359234391 x2_input_acceleration
   + 0.8514966533422488 x3_input_acceleration >= -1.059907629887777
 CBF_Constraint_obs1: - 0.4438691038912369 x1_input_acceleration
   - 0.3118918359234391 x2_input_acceleration
   + 0.8514966533422488 x3_input_acceleration >= -1.224178561697441
 CBF_Constraint_for_table: x3_input_acceleration >= -2.368716704308991
 Relaxed_CLF_constraint: - 1.41343e-08 x1_input_acceleration
   - 3.13498e-08 x2_input_acceleration
   + 0.4266670898318299 x3_input_acceleration + relaxation_CLF
   <= -20.07281792221822
 CBF_Constraint_obs2: - 0.1238662343773585 x1_input_acceleration
   - 0.3118938437104526 x2_input_acceleration
   + 0.8541472573849518 x3_input_acceleration >= -1.002735188826927
 CBF_Constraint_obs3: 0.2561337656226415 x1_input_acceleration
   - 0.3118938437104526 x2_input_acceleration
   + 0.8541472573849518 x3_input_acceleration >= -1.065562266168437
 CBF_Constraint_obs1: - 0.4437727742943507 x1_input_acceleration
   - 0.3118397226929965 x2_input_acceleration
   + 0.8541852851483185 x3_input_acceleration >= -1.22976348652258
 CBF_Constraint_for_table: x3_input_acceleration >= -2.375436197661598
 Relaxed_CLF_constraint: - 9.88887e-09 x1_input_acceleration
   - 2.19643e-08 x2_input_acceleration
   + 0.4279894116209427 x3_input_acceleration + relaxation_CLF
   <= -20.07326997458386
 CBF_Constraint_obs2: - 0.1238662345302618 x1_input_acceleration
   - 0.3118938311072985 x2_input_acceleration
   + 0.8567919009632063 x3_input_acceleration >= -1.008391209369548
 CBF_Constraint_obs3: 0.2561337654697382 x1_input_acceleration
   - 0.3118938311072985 x2_input_acceleration
   + 0.8567919009632063 x3_input_acceleration >= -1.0712182865658
 CBF_Constraint_obs1: - 0.4438662345302618 x1_input_acceleration
   - 0.3118938311072985 x2_input_acceleration
   + 0.8567919009632063 x3_input_acceleration >= -1.235484196993758
 CBF_Constraint_for_table: x3_input_acceleration >= -2.382047806607234
 Relaxed_CLF_constraint: - 9.96533e-09 x1_input_acceleration
   - 1.56627e-08 x2_input_acceleration
   + 0.42931173341007 x3_input_acceleration + relaxation_CLF
   <= -20.07372342577742
 CBF_Constraint_obs2: - 0.1238533613445122 x1_input_acceleration
   - 0.3118848964399199 x2_input_acceleration
   + 0.8594447699826593 x3_input_acceleration >= -1.014071445492775
 CBF_Constraint_obs3: 0.2561337640522163 x1_input_acceleration
   - 0.3118938371366362 x2_input_acceleration
   + 0.8594365445416803 x3_input_acceleration >= -1.076891806031393
 CBF_Constraint_obs1: - 0.4438662359477837 x1_input_acceleration
   - 0.3118938371366362 x2_input_acceleration
   + 0.8594365445416803 x3_input_acceleration >= -1.241157718940015
 CBF_Constraint_for_table: x3_input_acceleration >= -2.388580141375868
 Relaxed_CLF_constraint: - 1.06741e-08 x1_input_acceleration
   - 1.86774e-08 x2_input_acceleration
   + 0.4306340551993069 x3_input_acceleration + relaxation_CLF
   <= -20.07417827579896
 CBF_Constraint_obs2: - 0.1238533613623218 x1_input_acceleration
   - 0.3118848852447027 x2_input_acceleration
   + 0.8620894135608828 x3_input_acceleration >= -1.019762492173242
 CBF_Constraint_obs3: 0.2561466386376782 x1_input_acceleration
   - 0.3118848852447027 x2_input_acceleration
   + 0.8620894135608828 x3_input_acceleration >= -1.082601798879037
 CBF_Constraint_obs1: - 0.4438533613623218 x1_input_acceleration
   - 0.3118848852447027 x2_input_acceleration
   + 0.8620894135608828 x3_input_acceleration >= -1.2468451812631
 CBF_Constraint_for_table: x3_input_acceleration >= -2.395191750321427
 Relaxed_CLF_constraint: - 1.0683e-08 x1_input_acceleration
   - 1.30798e-08 x2_input_acceleration
   + 0.4319563769884187 x3_input_acceleration + relaxation_CLF
   <= -20.07463452464838
 CBF_Constraint_obs2: - 0.123788512535254 x1_input_acceleration
   - 0.3115149831393356 x2_input_acceleration
   + 0.8650249278057698 x3_input_acceleration >= -1.02579163285171
 CBF_Constraint_obs3: 0.256211487464746 x1_input_acceleration
   - 0.3115149831393356 x2_input_acceleration
   + 0.8650249278057698 x3_input_acceleration >= -1.088692545943219
 CBF_Constraint_obs1: - 0.4438533623887696 x1_input_acceleration
   - 0.3118848895648116 x2_input_acceleration
   + 0.8647340571392659 x3_input_acceleration >= -1.252553726524008
 CBF_Constraint_for_table: x3_input_acceleration >= -2.401441261050374
 Relaxed_CLF_constraint: - 1.11962e-08 x1_input_acceleration
   - 1.52399e-08 x2_input_acceleration
   + 0.4332786987776102 x3_input_acceleration + relaxation_CLF
   <= -20.07509217232577
 CBF_Constraint_obs2: - 0.1238667268401394 x1_input_acceleration
   - 0.3118879227989191 x2_input_acceleration
   + 0.8673585543515125 x3_input_acceleration >= -1.031159880265825
 CBF_Constraint_obs3: 0.2562114744538059 x1_input_acceleration
   - 0.3115150361412042 x2_input_acceleration
   + 0.8676695713879505 x3_input_acceleration >= -1.094420528120499
 CBF_Constraint_obs1: - 0.4437885255461941 x1_input_acceleration
   - 0.3115150361412042 x2_input_acceleration
   + 0.8676695713879505 x3_input_acceleration >= -1.258550447826338
 CBF_Constraint_for_table: x3_input_acceleration >= -2.408012040824155
 Relaxed_CLF_constraint: - 1.77017e-08 x1_input_acceleration
   - 4.17408e-08 x2_input_acceleration
   + 0.4346010205687006 x3_input_acceleration + relaxation_CLF
   <= -20.07555121883174
 CBF_Constraint_obs2: - 0.1238667164983429 x1_input_acceleration
   - 0.3118878990423348 x2_input_acceleration
   + 0.8700031979297851 x3_input_acceleration >= -1.036903236791469
 CBF_Constraint_obs3: 0.2561332835016571 x1_input_acceleration
   - 0.3118878990423348 x2_input_acceleration
   + 0.8700031979297851 x3_input_acceleration >= -1.099729856118043
 CBF_Constraint_obs1: - 0.4438667164983429 x1_input_acceleration
   - 0.3118878990423348 x2_input_acceleration
   + 0.8700031979297851 x3_input_acceleration >= -1.263996609990143
 CBF_Constraint_for_table: x3_input_acceleration >= -2.414623649769836
 Relaxed_CLF_constraint: - 1.25308e-08 x1_input_acceleration
   - 2.98625e-08 x2_input_acceleration
   + 0.4359233423578369 x3_input_acceleration + relaxation_CLF
   <= -20.07601166416497
 CBF_Constraint_obs2: - 0.123777064335739 x1_input_acceleration
   - 0.3115376473968827 x2_input_acceleration
   + 0.8729526596615229 x3_input_acceleration >= -1.043028516764106
 CBF_Constraint_obs3: 0.2561332902785188 x1_input_acceleration
   - 0.3118878842895829 x2_input_acceleration
   + 0.8726478415081416 x3_input_acceleration >= -1.105490712555439
 CBF_Constraint_obs1: - 0.4438667097214812 x1_input_acceleration
   - 0.3118878842895829 x2_input_acceleration
   + 0.8726478415081416 x3_input_acceleration >= -1.269757454568031
 CBF_Constraint_for_table: x3_input_acceleration >= -2.421235258715727
 Relaxed_CLF_constraint: - 9.14236e-09 x1_input_acceleration
   - 2.24861e-08 x2_input_acceleration
   + 0.4372456641470151 x3_input_acceleration + relaxation_CLF
   <= -20.07647350832615
 CBF_Constraint_obs2: - 0.1237770643687272 x1_input_acceleration
   - 0.3115376339642624 x2_input_acceleration
   + 0.8755973032397464 x3_input_acceleration >= -1.048808870600634
 CBF_Constraint_obs3: 0.2562229356312729 x1_input_acceleration
   - 0.3115376339642624 x2_input_acceleration
   + 0.8755973032397464 x3_input_acceleration >= -1.111720659450343
 CBF_Constraint_obs1: - 0.4437770643687272 x1_input_acceleration
   - 0.3115376339642624 x2_input_acceleration
   + 0.8755973032397464 x3_input_acceleration >= -1.275830522095616
 CBF_Constraint_for_table: x3_input_acceleration >= -2.428324002842713
 Relaxed_CLF_constraint: - 9.15885e-09 x1_input_acceleration
   - 1.57698e-08 x2_input_acceleration
   + 0.4385679859361268 x3_input_acceleration + relaxation_CLF
   <= -20.07693675131523
 CBF_Constraint_obs2: - 0.1238018710944943 x1_input_acceleration
   - 0.3118446423091377 x2_input_acceleration
   + 0.8779849315926644 x3_input_acceleration >= -1.054289405408984
 CBF_Constraint_obs3: 0.2561981289055057 x1_input_acceleration
   - 0.3118446423091377 x2_input_acceleration
   + 0.8779849315926644 x3_input_acceleration >= -1.117177627869215
 CBF_Constraint_obs1: - 0.4438018710944943 x1_input_acceleration
   - 0.3118446423091377 x2_input_acceleration
   + 0.8779849315926644 x3_input_acceleration >= -1.28133090228458
 CBF_Constraint_for_table: x3_input_acceleration >= -2.434881073545717
 Relaxed_CLF_constraint: - 1.47556e-08 x1_input_acceleration
   - 3.7982e-08 x2_input_acceleration
   + 0.4398903077267216 x3_input_acceleration + relaxation_CLF
   <= -20.07740139313276
 CBF_Constraint_obs2: - 0.1237055433665964 x1_input_acceleration
   - 0.311502379014033 x2_input_acceleration
   + 0.8809132932807262 x3_input_acceleration >= -1.060431279837152
 CBF_Constraint_obs3: 0.2562944566334036 x1_input_acceleration
   - 0.311502379014033 x2_input_acceleration
   + 0.8809132932807262 x3_input_acceleration >= -1.123411013638886
 CBF_Constraint_obs1: - 0.4437055433665964 x1_input_acceleration
   - 0.311502379014033 x2_input_acceleration
   + 0.8809132932807262 x3_input_acceleration >= -1.287395714530429
 CBF_Constraint_for_table: x3_input_acceleration >= -2.441124325788333
 Relaxed_CLF_constraint: - 1.14447e-08 x1_input_acceleration
   - 3.12696e-08 x2_input_acceleration
   + 0.4412126295161871 x3_input_acceleration + relaxation_CLF
   <= -20.07786743377784
 CBF_Constraint_obs2: - 0.1237055434963531 x1_input_acceleration
   - 0.3115023606801273 x2_input_acceleration
   + 0.8835579368589808 x3_input_acceleration >= -1.066264262484399
 CBF_Constraint_obs3: 0.2562944565036469 x1_input_acceleration
   - 0.3115023606801273 x2_input_acceleration
   + 0.8835579368589808 x3_input_acceleration >= -1.129243996162863
 CBF_Constraint_obs1: - 0.4437055434963531 x1_input_acceleration
   - 0.3115023606801273 x2_input_acceleration
   + 0.8835579368589808 x3_input_acceleration >= -1.293228697281481
 CBF_Constraint_for_table: x3_input_acceleration >= -2.447735934733969
 Relaxed_CLF_constraint: - 1.15095e-08 x1_input_acceleration
   - 2.21026e-08 x2_input_acceleration
   + 0.4425349513053143 x3_input_acceleration + relaxation_CLF
   <= -20.07833487325072
 CBF_Constraint_obs2: - 0.1238319063036395 x1_input_acceleration
   - 0.3119418760178545 x2_input_acceleration
   + 0.8858975238654723 x3_input_acceleration >= -1.071820622279134
 CBF_Constraint_obs3: 0.2561680936963605 x1_input_acceleration
   - 0.3119418760178545 x2_input_acceleration
   + 0.8858975238654723 x3_input_acceleration >= -1.134680311290676
 CBF_Constraint_obs1: - 0.4438319063036395 x1_input_acceleration
   - 0.3119418760178545 x2_input_acceleration
   + 0.8858975238654723 x3_input_acceleration >= -1.298886147322045
 CBF_Constraint_for_table: x3_input_acceleration >= -2.454292707406906
 Relaxed_CLF_constraint: - 1.19897e-08 x1_input_acceleration
   - 1.74462e-08 x2_input_acceleration
   + 0.4438572730945403 x3_input_acceleration + relaxation_CLF
   <= -20.07880371155157
 CBF_Constraint_obs2: - 0.1238319067948526 x1_input_acceleration
   - 0.311941867513098 x2_input_acceleration
   + 0.888542167443777 x3_input_acceleration >= -1.077686566467117
 CBF_Constraint_obs3: 0.2561680932051474 x1_input_acceleration
   - 0.311941867513098 x2_input_acceleration
   + 0.888542167443777 x3_input_acceleration >= -1.140546255012007
 CBF_Constraint_obs1: - 0.4438319067948526 x1_input_acceleration
   - 0.311941867513098 x2_input_acceleration
   + 0.888542167443777 x3_input_acceleration >= -1.304752091902999
 CBF_Constraint_for_table: x3_input_acceleration >= -2.460904316352667
 Relaxed_CLF_constraint: - 1.22353e-08 x1_input_acceleration
   - 1.31938e-08 x2_input_acceleration
   + 0.4451795948836927 x3_input_acceleration + relaxation_CLF
   <= -20.07927394868032
 CBF_Constraint_obs2: - 0.1238652859555945 x1_input_acceleration
   - 0.311882503602636 x2_input_acceleration
   + 0.8911627307456109 x3_input_acceleration >= -1.083480397235355
 CBF_Constraint_obs3: 0.2561680926454797 x1_input_acceleration
   - 0.3119418698288323 x2_input_acceleration
   + 0.8911868110221002 x3_input_acceleration >= -1.146429692477089
 CBF_Constraint_obs1: - 0.4438319073545203 x1_input_acceleration
   - 0.3119418698288323 x2_input_acceleration
   + 0.8911868110221002 x3_input_acceleration >= -1.3106355303475
 CBF_Constraint_for_table: x3_input_acceleration >= -2.467522481809401
 Relaxed_CLF_constraint: - 1.25152e-08 x1_input_acceleration
   - 1.43517e-08 x2_input_acceleration
   + 0.4465019166728543 x3_input_acceleration + relaxation_CLF
   <= -20.07974558463701
 CBF_Constraint_obs2: - 0.1236964839985997 x1_input_acceleration
   - 0.3115414612564211 x2_input_acceleration
   + 0.8940426934630235 x3_input_acceleration >= -1.089589049962525
 CBF_Constraint_obs3: 0.2563035160014003 x1_input_acceleration
   - 0.3115414612564211 x2_input_acceleration
   + 0.8940426934630235 x3_input_acceleration >= -1.152577390163855
 CBF_Constraint_obs1: - 0.4436964839985997 x1_input_acceleration
   - 0.3115414612564211 x2_input_acceleration
   + 0.8940426934630235 x3_input_acceleration >= -1.316546237161405
 CBF_Constraint_for_table: x3_input_acceleration >= -2.474133196689327
 Relaxed_CLF_constraint: - 1.17137e-08 x1_input_acceleration
   - 2.20673e-08 x2_input_acceleration
   + 0.4478242384627739 x3_input_acceleration + relaxation_CLF
   <= -20.08021861942191
 CBF_Constraint_obs2: - 0.1236964840442366 x1_input_acceleration
   - 0.3115414481251604 x2_input_acceleration
   + 0.8966873370412507 x3_input_acceleration >= -1.095508843093697
 CBF_Constraint_obs3: 0.2563035159557635 x1_input_acceleration
   - 0.3115414481251604 x2_input_acceleration
   + 0.8966873370412507 x3_input_acceleration >= -1.158497183251672
 CBF_Constraint_obs1: - 0.4436964840442366 x1_input_acceleration
   - 0.3115414481251604 x2_input_acceleration
   + 0.8966873370412507 x3_input_acceleration >= -1.322466030329086
 CBF_Constraint_for_table: x3_input_acceleration >= -2.480706062615791
 Relaxed_CLF_constraint: - 1.17365e-08 x1_input_acceleration
   - 1.55016e-08 x2_input_acceleration
   + 0.4491465602518875 x3_input_acceleration + relaxation_CLF
   <= -20.08069305303444
 CBF_Constraint_obs2: - 0.1238505031178723 x1_input_acceleration
   - 0.3118981250657131 x2_input_acceleration
   + 0.8990933236219243 x3_input_acceleration >= -1.10123524015447
 CBF_Constraint_obs3: 0.2561494968821277 x1_input_acceleration
   - 0.3118981250657131 x2_input_acceleration
   + 0.8990933236219243 x3_input_acceleration >= -1.164077262192491
 CBF_Constraint_obs1: - 0.4438505031178723 x1_input_acceleration
   - 0.3118981250657131 x2_input_acceleration
   + 0.8990933236219243 x3_input_acceleration >= -1.328315642648767
 CBF_Constraint_for_table: x3_input_acceleration >= -2.487317671561677
 Relaxed_CLF_constraint: - 1.20722e-08 x1_input_acceleration
   - 1.68747e-08 x2_input_acceleration
   + 0.4504688820410647 x3_input_acceleration + relaxation_CLF
   <= -20.08116888547493
 CBF_Constraint_obs2: - 0.1237870849563859 x1_input_acceleration
   - 0.3118380381755412 x2_input_acceleration
   + 0.9017994791939026 x3_input_acceleration >= -1.107260631411955
 CBF_Constraint_obs3: 0.2561494957015731 x1_input_acceleration
   - 0.3118981196574748 x2_input_acceleration
   + 0.9017379672004944 x3_input_acceleration >= -1.170030453356148
 CBF_Constraint_obs1: - 0.4438505042984269 x1_input_acceleration
   - 0.3118981196574748 x2_input_acceleration
   + 0.9017379672004944 x3_input_acceleration >= -1.334268835878395
 CBF_Constraint_for_table: x3_input_acceleration >= -2.493929280508102
 Relaxed_CLF_constraint: - 1.26625e-08 x1_input_acceleration
   - 1.41706e-08 x2_input_acceleration
   + 0.4517912038303497 x3_input_acceleration + relaxation_CLF
   <= -20.08164611674339
 CBF_Constraint_obs2: - 0.1237870773265652 x1_input_acceleration
   - 0.3118380296765866 x2_input_acceleration
   + 0.9044441227721292 x3_input_acceleration >= -1.113231710603344
 CBF_Constraint_obs3: 0.2562129226734348 x1_input_acceleration
   - 0.3118380296765866 x2_input_acceleration
   + 0.9044441227721292 x3_input_acceleration >= -1.176133987143107
 CBF_Constraint_obs1: - 0.4438504966686062 x1_input_acceleration
   - 0.3118981111585202 x2_input_acceleration
   + 0.904382610778721 x3_input_acceleration >= -1.340239502270195
 CBF_Constraint_for_table: x3_input_acceleration >= -2.50063476676919
 Relaxed_CLF_constraint: - 8.84761e-09 x1_input_acceleration
   - 9.92115e-09 x2_input_acceleration
   + 0.453113525619463 x3_input_acceleration + relaxation_CLF
   <= -20.08212474683972
 CBF_Constraint_obs2: - 0.1238609936395874 x1_input_acceleration
   - 0.311878824425605 x2_input_acceleration
   + 0.907029161707859 x3_input_acceleration >= -1.119139808823681
 CBF_Constraint_obs3: 0.2561390063604126 x1_input_acceleration
   - 0.311878824425605 x2_input_acceleration
   + 0.907029161707859 x3_input_acceleration >= -1.181971864866073
 CBF_Constraint_obs1: - 0.4438609936395874 x1_input_acceleration
   - 0.311878824425605 x2_input_acceleration
   + 0.907029161707859 x3_input_acceleration >= -1.346228603735351
 CBF_Constraint_for_table: x3_input_acceleration >= -2.507220447699976
 Relaxed_CLF_constraint: - 1.21244e-08 x1_input_acceleration
   - 2.25071e-08 x2_input_acceleration
   + 0.4544358474097156 x3_input_acceleration + relaxation_CLF
   <= -20.08260477576439
 CBF_Constraint_obs2: - 0.1238371518076766 x1_input_acceleration
   - 0.3119294748777308 x2_input_acceleration
   + 0.9097266150013585 x3_input_acceleration >= -1.125297689383955
 CBF_Constraint_obs3: 0.2561628481923234 x1_input_acceleration
   - 0.3119294748777308 x2_input_acceleration
   + 0.9097266150013585 x3_input_acceleration >= -1.188152395166663
 CBF_Constraint_obs1: - 0.4438609936655868 x1_input_acceleration
   - 0.3118788109296717 x2_input_acceleration
   + 0.9096738052860875 x3_input_acceleration >= -1.352234258035474
 CBF_Constraint_for_table: x3_input_acceleration >= -2.514258229855691
 Relaxed_CLF_constraint: - 1.21374e-08 x1_input_acceleration
   - 1.57592e-08 x2_input_acceleration
   + 0.4557581691988299 x3_input_acceleration + relaxation_CLF
   <= -20.08308620351659
 CBF_Constraint_obs2: - 0.1238371518514151 x1_input_acceleration
   - 0.3119294750545585 x2_input_acceleration
   + 0.9123712585796084 x3_input_acceleration >= -1.131321188835777
 CBF_Constraint_obs3: 0.2561628481485849 x1_input_acceleration
   - 0.3119294750545585 x2_input_acceleration
   + 0.9123712585796084 x3_input_acceleration >= -1.194175894576933
 CBF_Constraint_obs1: - 0.4438371518514151 x1_input_acceleration
   - 0.3119294750545585 x2_input_acceleration
   + 0.9123712585796084 x3_input_acceleration >= -1.358390910316909
 CBF_Constraint_for_table: x3_input_acceleration >= -2.520869838801316
 Relaxed_CLF_constraint: - 1.21592e-08 x1_input_acceleration
   - 1.58476e-08 x2_input_acceleration
   + 0.4570804909879548 x3_input_acceleration + relaxation_CLF
   <= -20.08356903009672
 CBF_Constraint_obs2: - 0.1237160365315187 x1_input_acceleration
   - 0.3114656369989208 x2_input_acceleration
   + 0.9152918716635277 x3_input_acceleration >= -1.137594638824419
 CBF_Constraint_obs3: 0.2561628468302977 x1_input_acceleration
   - 0.3119294803445629 x2_input_acceleration
   + 0.9150159021582176 x3_input_acceleration >= -1.20021688250831
 CBF_Constraint_obs1: - 0.4438371531697023 x1_input_acceleration
   - 0.3119294803445629 x2_input_acceleration
   + 0.9150159021582176 x3_input_acceleration >= -1.36443190055529
 CBF_Constraint_for_table: x3_input_acceleration >= -2.527039777330053
 Relaxed_CLF_constraint: - 1.28184e-08 x1_input_acceleration
   - 1.84926e-08 x2_input_acceleration
   + 0.4584028127772594 x3_input_acceleration + relaxation_CLF
   <= -20.08405325550484
 CBF_Constraint_obs2: - 0.1237160300668323 x1_input_acceleration
   - 0.3114656308236605 x2_input_acceleration
   + 0.9179365152422783 x3_input_acceleration >= -1.143654926618769
 CBF_Constraint_obs3: 0.2562839699331677 x1_input_acceleration
   - 0.3114656308236605 x2_input_acceleration
   + 0.9179365152422783 x3_input_acceleration >= -1.206624698055279
 CBF_Constraint_obs1: - 0.4437160300668324 x1_input_acceleration
   - 0.3114656308236605 x2_input_acceleration
   + 0.9179365152422783 x3_input_acceleration >= -1.370627750672235
 CBF_Constraint_for_table: x3_input_acceleration >= -2.533651386276929
 Relaxed_CLF_constraint: - 9.58603e-09 x1_input_acceleration
   - 1.5405e-08 x2_input_acceleration
   + 0.4597251345666347 x3_input_acceleration + relaxation_CLF
   <= -20.08453887974093
 CBF_Constraint_obs2: - 0.1237699127028817 x1_input_acceleration
   - 0.3115434746415133 x2_input_acceleration
   + 0.9205698339380235 x3_input_acceleration >= -1.149783933798583
 CBF_Constraint_obs3: 0.2562300872971183 x1_input_acceleration
   - 0.3115434746415133 x2_input_acceleration
   + 0.9205698339380235 x3_input_acceleration >= -1.212702516730845
 CBF_Constraint_obs1: - 0.4437699127028817 x1_input_acceleration
   - 0.3115434746415133 x2_input_acceleration
   + 0.9205698339380235 x3_input_acceleration >= -1.376799863960888
 CBF_Constraint_for_table: x3_input_acceleration >= -2.540224848249904
 Relaxed_CLF_constraint: - 9.60461e-09 x1_input_acceleration
   - 1.54809e-08 x2_input_acceleration
   + 0.461047456355761 x3_input_acceleration + relaxation_CLF
   <= -20.08502590280485
 CBF_Constraint_obs2: - 0.1237899399779173 x1_input_acceleration
   - 0.3115306005088312 x2_input_acceleration
   + 0.9231603564988522 x3_input_acceleration >= -1.155750385130272
 CBF_Constraint_obs3: 0.2562300871827273 x1_input_acceleration
   - 0.3115434751121027 x2_input_acceleration
   + 0.9232144775163083 x3_input_acceleration >= -1.218797707447879
 CBF_Constraint_obs1: - 0.4437699128172727 x1_input_acceleration
   - 0.3115434751121027 x2_input_acceleration
   + 0.9232144775163083 x3_input_acceleration >= -1.382895054878106
 CBF_Constraint_for_table: x3_input_acceleration >= -2.547251901569701
 Relaxed_CLF_constraint: - 9.66181e-09 x1_input_acceleration
   - 1.57161e-08 x2_input_acceleration
   + 0.4623697781449034 x3_input_acceleration + relaxation_CLF
   <= -20.08551432469671
 CBF_Constraint_obs2: - 0.1237899429260523 x1_input_acceleration
   - 0.3115306115975278 x2_input_acceleration
   + 0.925805 x3_input_acceleration >= -1.161862712627011
 CBF_Constraint_obs3: 0.2562100570739477 x1_input_acceleration
   - 0.3115306115975278 x2_input_acceleration
   + 0.925805 x3_input_acceleration >= -1.224762266847261
 CBF_Constraint_obs1: - 0.4437899429260523 x1_input_acceleration
   - 0.3115306115975278 x2_input_acceleration
   + 0.925805 x3_input_acceleration >= -1.388894666967853
 CBF_Constraint_for_table: x3_input_acceleration >= -2.553863510517865
 Relaxed_CLF_constraint: - 1.11359e-08 x1_input_acceleration
   - 2.12605e-08 x2_input_acceleration
   + 0.4636920999345361 x3_input_acceleration + relaxation_CLF
   <= -20.08600414541668
 CBF_Constraint_obs2: - 0.1236611991638527 x1_input_acceleration
   - 0.3115468184695118 x2_input_acceleration
   + 0.9285788665272439 x3_input_acceleration >= -1.168265279547672
 CBF_Constraint_obs3: 0.2563388008361474 x1_input_acceleration
   - 0.3115468184695118 x2_input_acceleration
   + 0.9285788665272439 x3_input_acceleration >= -1.231287140342012
 CBF_Constraint_obs1: - 0.4437899451965675 x1_input_acceleration
   - 0.3115306060061329 x2_input_acceleration
   + 0.9284496436573708 x3_input_acceleration >= -1.395024468429525
 CBF_Constraint_for_table: x3_input_acceleration >= -2.560090669507196
 Relaxed_CLF_constraint: - 1.22711e-08 x1_input_acceleration
   - 1.84648e-08 x2_input_acceleration
   + 0.4650144217241627 x3_input_acceleration + relaxation_CLF
   <= -20.08649536496458
 CBF_Constraint_obs2: - 0.1236611991845916 x1_input_acceleration
   - 0.3115468073985626 x2_input_acceleration
   + 0.931223510105474 x3_input_acceleration >= -1.17441341394646
 CBF_Constraint_obs3: 0.2563388008154084 x1_input_acceleration
   - 0.3115468073985626 x2_input_acceleration
   + 0.931223510105474 x3_input_acceleration >= -1.237435274721098
 CBF_Constraint_obs1: - 0.4436611991845916 x1_input_acceleration
   - 0.3115468073985626 x2_input_acceleration
   + 0.931223510105474 x3_input_acceleration >= -1.401342373294133
 CBF_Constraint_for_table: x3_input_acceleration >= -2.566684397059339
 Relaxed_CLF_constraint: - 1.22815e-08 x1_input_acceleration
   - 1.29293e-08 x2_input_acceleration
   + 0.4663367435132778 x3_input_acceleration + relaxation_CLF
   <= -20.08698798334023
 CBF_Constraint_obs2: - 0.1237475075796791 x1_input_acceleration
   - 0.3115733761419575 x2_input_acceleration
   + 0.9338255959705035 x3_input_acceleration >= -1.180527072552933
 CBF_Constraint_obs3: 0.2562524924203209 x1_input_acceleration
   - 0.3115733761419575 x2_input_acceleration
   + 0.9338255959705035 x3_input_acceleration >= -1.243466940352238
 CBF_Constraint_obs1: - 0.4437475075796791 x1_input_acceleration
   - 0.3115733761419575 x2_input_acceleration
   + 0.9338255959705035 x3_input_acceleration >= -1.407525078616676
 CBF_Constraint_for_table: x3_input_acceleration >= -2.573315079499165
 Relaxed_CLF_constraint: - 1.27162e-08 x1_input_acceleration
   - 5.46524e-09 x2_input_acceleration
   + 0.4676590653039773 x3_input_acceleration + relaxation_CLF
   <= -20.08748200054439
 CBF_Constraint_obs2: - 0.1238614716700448 x1_input_acceleration
   - 0.3118742604259586 x2_input_acceleration
   + 0.9361507586527351 x3_input_acceleration >= -1.186231826758249
 CBF_Constraint_obs3: 0.2561385283299552 x1_input_acceleration
   - 0.3118742604259586 x2_input_acceleration
   + 0.9361507586527351 x3_input_acceleration >= -1.249063428671707
 CBF_Constraint_obs1: - 0.4438614716700448 x1_input_acceleration
   - 0.3118742604259586 x2_input_acceleration
   + 0.9361507586527351 x3_input_acceleration >= -1.413321004094285
 CBF_Constraint_for_table: x3_input_acceleration >= -2.579926688444734
 Relaxed_CLF_constraint: - 1.2721e-08 x1_input_acceleration
   - 5.48382e-09 x2_input_acceleration
   + 0.4689813870930912 x3_input_acceleration + relaxation_CLF
   <= -20.0879774165759
 CBF_Constraint_obs2: - 0.1237818409587792 x1_input_acceleration
   - 0.3118286076129393 x2_input_acceleration
   + 0.9388234164148022 x3_input_acceleration >= -1.192435539857162
 CBF_Constraint_obs3: 0.2561385272358009 x1_input_acceleration
   - 0.3118742647708372 x2_input_acceleration
   + 0.9387954022317577 x3_input_acceleration >= -1.255261636765454
 CBF_Constraint_obs1: - 0.4438614727641991 x1_input_acceleration
   - 0.3118742647708372 x2_input_acceleration
   + 0.9387954022317577 x3_input_acceleration >= -1.419519214102802
 CBF_Constraint_for_table: x3_input_acceleration >= -2.586941224790972
 Relaxed_CLF_constraint: - 1.32681e-08 x1_input_acceleration
   - 7.65626e-09 x2_input_acceleration
   + 0.4703037088826025 x3_input_acceleration + relaxation_CLF
   <= -20.08847423143549
 CBF_Constraint_obs2: - 0.1238633804977531 x1_input_acceleration
   - 0.3119046647536119 x2_input_acceleration
   + 0.9413898586992662 x3_input_acceleration >= -1.198551903731523
 CBF_Constraint_obs3: 0.2562181586562996 x1_input_acceleration
   - 0.3118286092268785 x2_input_acceleration
   + 0.9414680599932115 x3_input_acceleration >= -1.261558667931219
 CBF_Constraint_obs1: - 0.4437818413437004 x1_input_acceleration
   - 0.3118286092268785 x2_input_acceleration
   + 0.9414680599932115 x3_input_acceleration >= -1.425676890282695
 CBF_Constraint_for_table: x3_input_acceleration >= -2.593224710167507
 Relaxed_CLF_constraint: - 1.34605e-08 x1_input_acceleration
   - 8.46323e-09 x2_input_acceleration
   + 0.4716260306718071 x3_input_acceleration + relaxation_CLF
   <= -20.0889724451229
 CBF_Constraint_obs2: - 0.1238633807632836 x1_input_acceleration
   - 0.3119046662967848 x2_input_acceleration
   + 0.9440345022783234 x3_input_acceleration >= -1.204784749304626
 CBF_Constraint_obs3: 0.2561366192367164 x1_input_acceleration
   - 0.3119046662967848 x2_input_acceleration
   + 0.9440345022783234 x3_input_acceleration >= -1.267614537579506
 CBF_Constraint_obs1: - 0.4438633807632836 x1_input_acceleration
   - 0.3119046662967848 x2_input_acceleration
   + 0.9440345022783234 x3_input_acceleration >= -1.431875453915253
 CBF_Constraint_for_table: x3_input_acceleration >= -2.59983631911515
 Relaxed_CLF_constraint: - 1.35933e-08 x1_input_acceleration
   - 9.23482e-09 x2_input_acceleration
   + 0.4729483524613358 x3_input_acceleration + relaxation_CLF
   <= -20.08947205763836
 CBF_Constraint_obs2: - 0.1237727817608702 x1_input_acceleration
   - 0.3118725992391795 x2_input_acceleration
   + 0.9467389889199189 x3_input_acceleration >= -1.21112366600267
 CBF_Constraint_obs3: 0.2562272182391298 x1_input_acceleration
   - 0.3118725992391795 x2_input_acceleration
   + 0.9467389889199189 x3_input_acceleration >= -1.274039523329844
 CBF_Constraint_obs1: - 0.4437727817608702 x1_input_acceleration
   - 0.3118725992391795 x2_input_acceleration
   + 0.9467389889199189 x3_input_acceleration >= -1.438141891411366
 CBF_Constraint_for_table: x3_input_acceleration >= -2.606768899072868
 Relaxed_CLF_constraint: - 1.36221e-08 x1_input_acceleration
   - 9.35546e-09 x2_input_acceleration
   + 0.4742706742504563 x3_input_acceleration + relaxation_CLF
   <= -20.08997306898159
 CBF_Constraint_obs2: - 0.1237727817957482 x1_input_acceleration
   - 0.3118725994089622 x2_input_acceleration
   + 0.9493836324981828 x3_input_acceleration >= -1.217391876788757
 CBF_Constraint_obs3: 0.2562272182042518 x1_input_acceleration
   - 0.3118725994089622 x2_input_acceleration
   + 0.9493836324981828 x3_input_acceleration >= -1.280307734082796
 CBF_Constraint_obs1: - 0.4437727817957483 x1_input_acceleration
   - 0.3118725994089622 x2_input_acceleration
   + 0.9493836324981828 x3_input_acceleration >= -1.444410102225355
 CBF_Constraint_for_table: x3_input_acceleration >= -2.613380508018528
 Relaxed_CLF_constraint: - 1.36396e-08 x1_input_acceleration
   - 9.44035e-09 x2_input_acceleration
   + 0.4755929960395882 x3_input_acceleration + relaxation_CLF
   <= -20.09047547915277
 CBF_Constraint_obs2: - 0.1238209460559488 x1_input_acceleration
   - 0.3119305452479457 x2_input_acceleration
   + 0.9520378128240417 x3_input_acceleration >= -1.22376036098457
 CBF_Constraint_obs3: 0.2562272144970297 x1_input_acceleration
   - 0.311872609533224 x2_input_acceleration
   + 0.9520282760808776 x3_input_acceleration >= -1.286593435604372
 CBF_Constraint_obs1: - 0.4437727855029703 x1_input_acceleration
   - 0.311872609533224 x2_input_acceleration
   + 0.9520282760808776 x3_input_acceleration >= -1.45069581023457
 CBF_Constraint_for_table: x3_input_acceleration >= -2.620057980107742
 Relaxed_CLF_constraint: - 1.54932e-08 x1_input_acceleration
   - 1.45025e-08 x2_input_acceleration
   + 0.4769153178309356 x3_input_acceleration + relaxation_CLF
   <= -20.09097928815271
 CBF_Constraint_obs2: - 0.1237069726227686 x1_input_acceleration
   - 0.3114962570659119 x2_input_acceleration
   + 0.9549822677654887 x3_input_acceleration >= -1.230405581235098
 CBF_Constraint_obs3: 0.2562930273772314 x1_input_acceleration
   - 0.3114962570659119 x2_input_acceleration
   + 0.9549822677654887 x3_input_acceleration >= -1.293383957243468
 CBF_Constraint_obs1: - 0.4437069726227686 x1_input_acceleration
   - 0.3114962570659119 x2_input_acceleration
   + 0.9549822677654887 x3_input_acceleration >= -1.457371159333313
 CBF_Constraint_for_table: x3_input_acceleration >= -2.626693430911219
 Relaxed_CLF_constraint: - 1.0817e-08 x1_input_acceleration
   - 1.01324e-08 x2_input_acceleration
   + 0.478237639620049 x3_input_acceleration + relaxation_CLF
   <= -20.09148449597974
 CBF_Constraint_obs2: - 0.1237603783898269 x1_input_acceleration
   - 0.3118240823737581 x2_input_acceleration
   + 0.9573349677936751 x3_input_acceleration >= -1.236301412709933
 CBF_Constraint_obs3: 0.2562396216101731 x1_input_acceleration
   - 0.3118240823737581 x2_input_acceleration
   + 0.9573349677936751 x3_input_acceleration >= -1.299229053239598
 CBF_Constraint_obs1: - 0.4437069726281082 x1_input_acceleration
   - 0.3114962568274935 x2_input_acceleration
   + 0.957626911343785 x3_input_acceleration >= -1.463693871132723
 CBF_Constraint_for_table: x3_input_acceleration >= -2.632856514905025
 Relaxed_CLF_constraint: - 1.08197e-08 x1_input_acceleration
   - 1.00132e-08 x2_input_acceleration
   + 0.4795599614091972 x3_input_acceleration + relaxation_CLF
   <= -20.09199110263472
 CBF_Constraint_obs2: - 0.1237603792063429 x1_input_acceleration
   - 0.3118240850180178 x2_input_acceleration
   + 0.9599796113727297 x3_input_acceleration >= -1.242639682137449
 CBF_Constraint_obs3: 0.2562396207936571 x1_input_acceleration
   - 0.3118240850180178 x2_input_acceleration
   + 0.9599796113727297 x3_input_acceleration >= -1.305567321891423
 CBF_Constraint_obs1: - 0.4437603792063429 x1_input_acceleration
   - 0.3118240850180178 x2_input_acceleration
   + 0.9599796113727297 x3_input_acceleration >= -1.469647985502523
 CBF_Constraint_for_table: x3_input_acceleration >= -2.639468123852661
 Relaxed_CLF_constraint: - 1.12279e-08 x1_input_acceleration
   - 1.13353e-08 x2_input_acceleration
   + 0.4808822831987244 x3_input_acceleration + relaxation_CLF
   <= -20.09249910811777
 CBF_Constraint_obs2: - 0.1237947130481929 x1_input_acceleration
   - 0.3115415646527009 x2_input_acceleration
   + 0.9628570706956272 x3_input_acceleration >= -1.249346270091877
 CBF_Constraint_obs3: 0.2562052869518071 x1_input_acceleration
   - 0.3115415646527009 x2_input_acceleration
   + 0.9628570706956272 x3_input_acceleration >= -1.312241292696094
 CBF_Constraint_obs1: - 0.4437947130481929 x1_input_acceleration
   - 0.3115415646527009 x2_input_acceleration
   + 0.9628570706956272 x3_input_acceleration >= -1.476382040530431
 CBF_Constraint_for_table: x3_input_acceleration >= -2.646049930481285
 Relaxed_CLF_constraint: - 1.20112e-08 x1_input_acceleration
   - 1.41608e-08 x2_input_acceleration
   + 0.4822046049889269 x3_input_acceleration + relaxation_CLF
   <= -20.09300851242901
 CBF_Constraint_obs2: - 0.12342611793577 x1_input_acceleration
   - 0.3117413594647671 x2_input_acceleration
   + 0.9653904920067855 x3_input_acceleration >= -1.255494354808424
 CBF_Constraint_obs3: 0.25657388206423 x1_input_acceleration
   - 0.3117413594647671 x2_input_acceleration
   + 0.9653904920067855 x3_input_acceleration >= -1.318739542769442
 CBF_Constraint_obs1: - 0.44342611793577 x1_input_acceleration
   - 0.3117413594647671 x2_input_acceleration
   + 0.9653904920067855 x3_input_acceleration >= -1.48223524915704
 CBF_Constraint_for_table: x3_input_acceleration >= -2.652657367101924
 Relaxed_CLF_constraint: - 1.20166e-08 x1_input_acceleration
   - 1.41822e-08 x2_input_acceleration
   + 0.4835269267780815 x3_input_acceleration + relaxation_CLF
   <= -20.09351931556778
 CBF_Constraint_obs2: - 0.1238228466205765 x1_input_acceleration
   - 0.3119497381166476 x2_input_acceleration
   + 0.9678805211371608 x3_input_acceleration >= -1.261796799566262
 CBF_Constraint_obs3: 0.2561771533794235 x1_input_acceleration
   - 0.3119497381166476 x2_input_acceleration
   + 0.9678805211371608 x3_input_acceleration >= -1.324665095276715
 CBF_Constraint_obs1: - 0.4438228466205765 x1_input_acceleration
   - 0.3119497381166476 x2_input_acceleration
   + 0.9678805211371608 x3_input_acceleration >= -1.488855076862723
 CBF_Constraint_for_table: x3_input_acceleration >= -2.65927195628147
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.4848492485675428 x3_input_acceleration + relaxation_CLF
   <= -20.0940315175346
 CBF_Constraint_obs2: - 0.1238228466205765 x1_input_acceleration
   - 0.3119497381166476 x2_input_acceleration
   + 0.9705251647153808 x3_input_acceleration >= -1.268204789752606
 CBF_Constraint_obs3: 0.2561771533794235 x1_input_acceleration
   - 0.3119497381166476 x2_input_acceleration
   + 0.9705251647153808 x3_input_acceleration >= -1.331073085463058
 CBF_Constraint_obs1: - 0.4438228466205765 x1_input_acceleration
   - 0.3119497381166476 x2_input_acceleration
   + 0.9705251647153808 x3_input_acceleration >= -1.495263067049067
 CBF_Constraint_for_table: x3_input_acceleration >= -2.66588356522702
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.4861715703566528 x3_input_acceleration + relaxation_CLF
   <= -20.09454511832922
 CBF_Constraint_obs2: - 0.1237413074665238 x1_input_acceleration
   - 0.3115784011796969 x2_input_acceleration
   + 0.9734966801655489 x3_input_acceleration >= -1.275110996935708
 CBF_Constraint_obs3: 0.2562586925334762 x1_input_acceleration
   - 0.3115784011796969 x2_input_acceleration
   + 0.9734966801655489 x3_input_acceleration >= -1.33805675484251
 CBF_Constraint_obs1: - 0.4437413074665238 x1_input_acceleration
   - 0.3115784011796969 x2_input_acceleration
   + 0.9734966801655489 x3_input_acceleration >= -1.502104042908927
 CBF_Constraint_for_table: x3_input_acceleration >= -2.672894823315788
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.4874938921457627 x3_input_acceleration + relaxation_CLF
   <= -20.09506011795177
 CBF_Constraint_obs2: - 0.1237775470905472 x1_input_acceleration
   - 0.3115408502534884 x2_input_acceleration
   + 0.9760569235667669 x3_input_acceleration >= -1.281332125728801
 CBF_Constraint_obs3: 0.2562224529094528 x1_input_acceleration
   - 0.3115408502534884 x2_input_acceleration
   + 0.9760569235667669 x3_input_acceleration >= -1.344243455992781
 CBF_Constraint_obs1: - 0.4437775470905472 x1_input_acceleration
   - 0.3115408502534884 x2_input_acceleration
   + 0.9760569235667669 x3_input_acceleration >= -1.508354163401239
 CBF_Constraint_for_table: x3_input_acceleration >= -2.679104100909104
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.4888162139348727 x3_input_acceleration + relaxation_CLF
   <= -20.09557651640225
 CBF_Constraint_obs2: - 0.1237775470905472 x1_input_acceleration
   - 0.3115408502534884 x2_input_acceleration
   + 0.9787015671449868 x3_input_acceleration >= -1.28779417509059
 CBF_Constraint_obs3: 0.2562224529094528 x1_input_acceleration
   - 0.3115408502534884 x2_input_acceleration
   + 0.9787015671449868 x3_input_acceleration >= -1.35070550535457
 CBF_Constraint_obs1: - 0.4437775470905472 x1_input_acceleration
   - 0.3115408502534884 x2_input_acceleration
   + 0.9787015671449868 x3_input_acceleration >= -1.514816212763028
 CBF_Constraint_for_table: x3_input_acceleration >= -2.685706471134714
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.4901385357239826 x3_input_acceleration + relaxation_CLF
   <= -20.09609431368066
 CBF_Constraint_obs2: - 0.1238161709003617 x1_input_acceleration
   - 0.3119409166292209 x2_input_acceleration
   + 0.9811211435845348 x3_input_acceleration >= -1.294045347540664
 CBF_Constraint_obs3: 0.2561838290996383 x1_input_acceleration
   - 0.3119409166292209 x2_input_acceleration
   + 0.9811211435845348 x3_input_acceleration >= -1.356919985185321
 CBF_Constraint_obs1: - 0.4437775470905472 x1_input_acceleration
   - 0.3115408502534884 x2_input_acceleration
   + 0.9813462107232067 x3_input_acceleration >= -1.521295747473957
 CBF_Constraint_for_table: x3_input_acceleration >= -2.692318080080264
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.4914608575130926 x3_input_acceleration + relaxation_CLF
   <= -20.096613509787
 CBF_Constraint_obs2: - 0.1238161709003617 x1_input_acceleration
   - 0.3119409166292209 x2_input_acceleration
   + 0.9837657871627548 x3_input_acceleration >= -1.300540879544826
 CBF_Constraint_obs3: 0.2561838290996383 x1_input_acceleration
   - 0.3119409166292209 x2_input_acceleration
   + 0.9837657871627548 x3_input_acceleration >= -1.363415517189482
 CBF_Constraint_obs1: - 0.4438161709003617 x1_input_acceleration
   - 0.3119409166292209 x2_input_acceleration
   + 0.9837657871627548 x3_input_acceleration >= -1.527593816265115
 CBF_Constraint_for_table: x3_input_acceleration >= -2.698921046352321
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.4927831793022026 x3_input_acceleration + relaxation_CLF
   <= -20.09713410472127
 CBF_Constraint_obs2: - 0.1234456684284378 x1_input_acceleration
   - 0.3117437444643039 x2_input_acceleration
   + 0.9865370310064776 x3_input_acceleration >= -1.307097886016806
 CBF_Constraint_obs3: 0.2565543315715622 x1_input_acceleration
   - 0.3117437444643039 x2_input_acceleration
   + 0.9865370310064776 x3_input_acceleration >= -1.37032450100979
 CBF_Constraint_obs1: - 0.4438161709003617 x1_input_acceleration
   - 0.3119409166292209 x2_input_acceleration
   + 0.9864104307409747 x3_input_acceleration >= -1.534106833618416
 CBF_Constraint_for_table: x3_input_acceleration >= -2.705560967504139
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.4941055010913125 x3_input_acceleration + relaxation_CLF
   <= -20.09765609848348
 CBF_Constraint_obs2: - 0.1237684871845414 x1_input_acceleration
   - 0.3118394695238131 x2_input_acceleration
   + 0.9890696178525198 x3_input_acceleration >= -1.313526502664716
 CBF_Constraint_obs3: 0.2562315128154586 x1_input_acceleration
   - 0.3118394695238131 x2_input_acceleration
   + 0.9890696178525198 x3_input_acceleration >= -1.376446439839402
 CBF_Constraint_obs1: - 0.4437684871845414 x1_input_acceleration
   - 0.3118394695238131 x2_input_acceleration
   + 0.9890696178525198 x3_input_acceleration >= -1.540541292412349
 CBF_Constraint_for_table: x3_input_acceleration >= -2.712172576449689
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.4954278228804225 x3_input_acceleration + relaxation_CLF
   <= -20.09817949107362
 CBF_Constraint_obs2: - 0.1237913753681351 x1_input_acceleration
   - 0.3118366085008639 x2_input_acceleration
   + 0.9917341693820947 x3_input_acceleration >= -1.320128797171061
 CBF_Constraint_obs3: 0.2562086246318649 x1_input_acceleration
   - 0.3118366085008639 x2_input_acceleration
   + 0.9917341693820947 x3_input_acceleration >= -1.383026990571333
 CBF_Constraint_obs1: - 0.4437913753681351 x1_input_acceleration
   - 0.3118366085008639 x2_input_acceleration
   + 0.9917341693820947 x3_input_acceleration >= -1.547161897465569
 CBF_Constraint_for_table: x3_input_acceleration >= -2.718796106324194
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.4967501446695324 x3_input_acceleration + relaxation_CLF
   <= -20.09870428249168
 CBF_Constraint_obs2: - 0.1237913753681351 x1_input_acceleration
   - 0.3118366085008639 x2_input_acceleration
   + 0.9943788129603146 x3_input_acceleration >= -1.326694498351525
 CBF_Constraint_obs3: 0.2562086246318649 x1_input_acceleration
   - 0.3118366085008639 x2_input_acceleration
   + 0.9943788129603146 x3_input_acceleration >= -1.389592691751796
 CBF_Constraint_obs1: - 0.4437913753681351 x1_input_acceleration
   - 0.3118366085008639 x2_input_acceleration
   + 0.9943788129603146 x3_input_acceleration >= -1.553727598646033
 CBF_Constraint_for_table: x3_input_acceleration >= -2.725407715269744
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.4980724664586424 x3_input_acceleration + relaxation_CLF
   <= -20.09923047273768
 CBF_Constraint_obs2: - 0.1238767292194535 x1_input_acceleration
   - 0.3118894182161349 x2_input_acceleration
   + 0.9969214133866791 x3_input_acceleration >= -1.333090947132753
 CBF_Constraint_obs3: 0.2562086246318649 x1_input_acceleration
   - 0.3118366085008639 x2_input_acceleration
   + 0.9970234565385345 x3_input_acceleration >= -1.3961758782814
 CBF_Constraint_obs1: - 0.4437913753681351 x1_input_acceleration
   - 0.3118366085008639 x2_input_acceleration
   + 0.9970234565385345 x3_input_acceleration >= -1.560310785175636
 CBF_Constraint_for_table: x3_input_acceleration >= -2.732019324215293
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.4993947882477524 x3_input_acceleration + relaxation_CLF
   <= -20.09975806181161
 CBF_Constraint_obs2: - 0.1237446453266312 x1_input_acceleration
   - 0.3115916334108371 x2_input_acceleration
   + 0.9999362018089542 x3_input_acceleration >= -1.340343113683456
 CBF_Constraint_obs3: 0.2562553546733688 x1_input_acceleration
   - 0.3115916334108371 x2_input_acceleration
   + 0.9999362018089542 x3_input_acceleration >= -1.403285700623156
 CBF_Constraint_obs1: - 0.4437446453266312 x1_input_acceleration
   - 0.3115916334108371 x2_input_acceleration
   + 0.9999362018089542 x3_input_acceleration >= -1.567338829944761
 CBF_Constraint_for_table: x3_input_acceleration >= -2.738624078626694
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5007171100368624 x3_input_acceleration + relaxation_CLF
   <= -20.10028704971347
 CBF_Constraint_obs2: - 0.1238552715473343 x1_input_acceleration
   - 0.3118863187746066 x2_input_acceleration
   + 1.002242052586271 x3_input_acceleration >= -1.346377920126476
 CBF_Constraint_obs3: 0.2561447284526657 x1_input_acceleration
   - 0.3118863187746066 x2_input_acceleration
   + 1.002242052586271 x3_input_acceleration >= -1.409215412156508
 CBF_Constraint_obs1: - 0.4437446453266312 x1_input_acceleration
   - 0.3115916334108371 x2_input_acceleration
   + 1.002580845387174 x3_input_acceleration >= -1.57395875975619
 CBF_Constraint_for_table: x3_input_acceleration >= -2.745182639438394
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5020394318259723 x3_input_acceleration + relaxation_CLF
   <= -20.10081743644326
 CBF_Constraint_obs2: - 0.1238552715473343 x1_input_acceleration
   - 0.3118863187746066 x2_input_acceleration
   + 1.004886696164491 x3_input_acceleration >= -1.353013095321531
 CBF_Constraint_obs3: 0.2561447284526657 x1_input_acceleration
   - 0.3118863187746066 x2_input_acceleration
   + 1.004886696164491 x3_input_acceleration >= -1.415850587351563
 CBF_Constraint_obs1: - 0.4438552715473343 x1_input_acceleration
   - 0.3118863187746066 x2_input_acceleration
   + 1.004886696164491 x3_input_acceleration >= -1.580097312559399
 CBF_Constraint_for_table: x3_input_acceleration >= -2.751794248383943
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5033617536150823 x3_input_acceleration + relaxation_CLF
   <= -20.10134922200098
 CBF_Constraint_obs2: - 0.1233545925312211 x1_input_acceleration
   - 0.3114092431978244 x2_input_acceleration
   + 1.007987076856663 x3_input_acceleration >= -1.360287524197032
 CBF_Constraint_obs3: 0.256645407468779 x1_input_acceleration
   - 0.3114092431978244 x2_input_acceleration
   + 1.007987076856663 x3_input_acceleration >= -1.423600661292372
 CBF_Constraint_obs1: - 0.4438552715473343 x1_input_acceleration
   - 0.3118863187746066 x2_input_acceleration
   + 1.007531339742711 x3_input_acceleration >= -1.586749973103593
 CBF_Constraint_for_table: x3_input_acceleration >= -2.758428805117732
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5046840754041922 x3_input_acceleration + relaxation_CLF
   <= -20.10188240638663
 CBF_Constraint_obs2: - 0.1234218265705277 x1_input_acceleration
   - 0.3117525659517306 x2_input_acceleration
   + 1.010363499533394 x3_input_acceleration >= -1.366571263551075
 CBF_Constraint_obs3: 0.2565781734294723 x1_input_acceleration
   - 0.3117525659517306 x2_input_acceleration
   + 1.010363499533394 x3_input_acceleration >= -1.429820528309074
 CBF_Constraint_obs1: - 0.4434218265705277 x1_input_acceleration
   - 0.3117525659517306 x2_input_acceleration
   + 1.010363499533394 x3_input_acceleration >= -1.593308724807497
 CBF_Constraint_for_table: x3_input_acceleration >= -2.765040414063282
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5060063971933022 x3_input_acceleration + relaxation_CLF
   <= -20.10241698960022
 CBF_Constraint_obs2: - 0.1238795902424027 x1_input_acceleration
   - 0.311897882075693 x2_input_acceleration
   + 1.012793447181133 x3_input_acceleration >= -1.372971260468713
 CBF_Constraint_obs3: 0.2565781734294723 x1_input_acceleration
   - 0.3117525659517306 x2_input_acceleration
   + 1.013008143111614 x3_input_acceleration >= -1.436509399335415
 CBF_Constraint_obs1: - 0.4434218265705277 x1_input_acceleration
   - 0.3117525659517306 x2_input_acceleration
   + 1.013008143111614 x3_input_acceleration >= -1.599997595833839
 CBF_Constraint_for_table: x3_input_acceleration >= -2.772059718779095
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5073287189824122 x3_input_acceleration + relaxation_CLF
   <= -20.10295297164173
 CBF_Constraint_obs2: - 0.1238795902424027 x1_input_acceleration
   - 0.311897882075693 x2_input_acceleration
   + 1.015438090759353 x3_input_acceleration >= -1.37967619735866
 CBF_Constraint_obs3: 0.2561204097575973 x1_input_acceleration
   - 0.311897882075693 x2_input_acceleration
   + 1.015438090759353 x3_input_acceleration >= -1.442490586628377
 CBF_Constraint_obs1: - 0.4438795902424027 x1_input_acceleration
   - 0.311897882075693 x2_input_acceleration
   + 1.015438090759353 x3_input_acceleration >= -1.606779869552582
 CBF_Constraint_for_table: x3_input_acceleration >= -2.778671327724645
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5086510407715221 x3_input_acceleration + relaxation_CLF
   <= -20.10349035251118
 CBF_Constraint_obs2: - 0.1237799312763382 x1_input_acceleration
   - 0.3115291677431125 x2_input_acceleration
   + 1.018432494393115 x3_input_acceleration >= -1.386970799221591
 CBF_Constraint_obs3: 0.2562200687236618 x1_input_acceleration
   - 0.3115291677431125 x2_input_acceleration
   + 1.018432494393115 x3_input_acceleration >= -1.44987986450907
 CBF_Constraint_obs1: - 0.4437799312763382 x1_input_acceleration
   - 0.3115291677431125 x2_input_acceleration
   + 1.018432494393115 x3_input_acceleration >= -1.613994744242662
 CBF_Constraint_for_table: x3_input_acceleration >= -2.784906831361662
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5099733625606321 x3_input_acceleration + relaxation_CLF
   <= -20.10402913220856
 CBF_Constraint_obs2: - 0.1237041141681839 x1_input_acceleration
   - 0.3115041337923068 x2_input_acceleration
   + 1.021082860017233 x3_input_acceleration >= -1.393684675316003
 CBF_Constraint_obs3: 0.2562958858318161 x1_input_acceleration
   - 0.3115041337923068 x2_input_acceleration
   + 1.021082860017233 x3_input_acceleration >= -1.456665766856229
 CBF_Constraint_obs1: - 0.4437041141681839 x1_input_acceleration
   - 0.3115041337923068 x2_input_acceleration
   + 1.021082860017233 x3_input_acceleration >= -1.620647966650551
 CBF_Constraint_for_table: x3_input_acceleration >= -2.791461517871451
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.511295684349742 x3_input_acceleration + relaxation_CLF
   <= -20.10456931073387
 CBF_Constraint_obs2: - 0.1237041141681839 x1_input_acceleration
   - 0.3115041337923068 x2_input_acceleration
   + 1.023727503595453 x3_input_acceleration >= -1.40044441856201
 CBF_Constraint_obs3: 0.2562958858318161 x1_input_acceleration
   - 0.3115041337923068 x2_input_acceleration
   + 1.023727503595453 x3_input_acceleration >= -1.463425510102236
 CBF_Constraint_obs1: - 0.4437041141681839 x1_input_acceleration
   - 0.3115041337923068 x2_input_acceleration
   + 1.023727503595453 x3_input_acceleration >= -1.627407709896558
 CBF_Constraint_for_table: x3_input_acceleration >= -2.798073126817001
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.512618006138852 x3_input_acceleration + relaxation_CLF
   <= -20.10511088808711
 CBF_Constraint_obs2: - 0.1237589504413773 x1_input_acceleration
   - 0.3115905605272311 x2_input_acceleration
   + 1.026260448069364 x3_input_acceleration >= -1.407019328119446
 CBF_Constraint_obs3: 0.2562410495586227 x1_input_acceleration
   - 0.3115905605272311 x2_input_acceleration
   + 1.026260448069364 x3_input_acceleration >= -1.469948325200138
 CBF_Constraint_obs1: - 0.4437589504413773 x1_input_acceleration
   - 0.3115905605272311 x2_input_acceleration
   + 1.026260448069364 x3_input_acceleration >= -1.634026488472548
 CBF_Constraint_for_table: x3_input_acceleration >= -2.805119849669411
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.513940327927962 x3_input_acceleration + relaxation_CLF
   <= -20.10565386426828
 CBF_Constraint_obs2: - 0.1237165119342972 x1_input_acceleration
   - 0.3114920936540622 x2_input_acceleration
   + 1.029012737636048 x3_input_acceleration >= -1.414000392439268
 CBF_Constraint_obs3: 0.2562410495586227 x1_input_acceleration
   - 0.3115905605272311 x2_input_acceleration
   + 1.028905091647584 x3_input_acceleration >= -1.476742300633626
 CBF_Constraint_obs1: - 0.4437589504413773 x1_input_acceleration
   - 0.3115905605272311 x2_input_acceleration
   + 1.028905091647584 x3_input_acceleration >= -1.640820463906037
 CBF_Constraint_for_table: x3_input_acceleration >= -2.811329723309175
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5152626497170719 x3_input_acceleration + relaxation_CLF
   <= -20.10619823927738
 CBF_Constraint_obs2: - 0.1238743450336625 x1_input_acceleration
   - 0.3118746362642306 x2_input_acceleration
   + 1.031353993572361 x3_input_acceleration >= -1.420377127700154
 CBF_Constraint_obs3: 0.2561256549663375 x1_input_acceleration
   - 0.3118746362642306 x2_input_acceleration
   + 1.031353993572361 x3_input_acceleration >= -1.483196499918175
 CBF_Constraint_obs1: - 0.4438743450336625 x1_input_acceleration
   - 0.3118746362642306 x2_input_acceleration
   + 1.031353993572361 x3_input_acceleration >= -1.647476603727084
 CBF_Constraint_for_table: x3_input_acceleration >= -2.817907357607203
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5165849715061819 x3_input_acceleration + relaxation_CLF
   <= -20.10674401311442
 CBF_Constraint_obs2: - 0.1238743450336625 x1_input_acceleration
   - 0.3118746362642306 x2_input_acceleration
   + 1.033998637150581 x3_input_acceleration >= -1.427204779664655
 CBF_Constraint_obs3: 0.2561256549663375 x1_input_acceleration
   - 0.3118746362642306 x2_input_acceleration
   + 1.033998637150581 x3_input_acceleration >= -1.490024151882676
 CBF_Constraint_obs1: - 0.4438743450336625 x1_input_acceleration
   - 0.3118746362642306 x2_input_acceleration
   + 1.033998637150581 x3_input_acceleration >= -1.654304255691585
 CBF_Constraint_for_table: x3_input_acceleration >= -2.824518966552753
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5179072932952918 x3_input_acceleration + relaxation_CLF
   <= -20.10729118577938
 CBF_Constraint_obs2: - 0.1237999584369828 x1_input_acceleration
   - 0.3118499599412936 x2_input_acceleration
   + 1.036645784123882 x3_input_acceleration >= -1.434014136207753
 CBF_Constraint_obs3: 0.2562000415630172 x1_input_acceleration
   - 0.3118499599412936 x2_input_acceleration
   + 1.036645784123882 x3_input_acceleration >= -1.496904175692619
 CBF_Constraint_obs1: - 0.4437999584369828 x1_input_acceleration
   - 0.3118499599412936 x2_input_acceleration
   + 1.036645784123882 x3_input_acceleration >= -1.66105410295734
 CBF_Constraint_for_table: x3_input_acceleration >= -2.831572841962536
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5192296150844018 x3_input_acceleration + relaxation_CLF
   <= -20.10783975727228
 CBF_Constraint_obs2: - 0.1237999584369828 x1_input_acceleration
   - 0.3118499599412936 x2_input_acceleration
   + 1.039290427702102 x3_input_acceleration >= -1.440876775422003
 CBF_Constraint_obs3: 0.2562000415630172 x1_input_acceleration
   - 0.3118499599412936 x2_input_acceleration
   + 1.039290427702102 x3_input_acceleration >= -1.503766814906869
 CBF_Constraint_obs1: - 0.4437999584369828 x1_input_acceleration
   - 0.3118499599412936 x2_input_acceleration
   + 1.039290427702102 x3_input_acceleration >= -1.667916742171589
 CBF_Constraint_for_table: x3_input_acceleration >= -2.838184450908086
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5205519368735118 x3_input_acceleration + relaxation_CLF
   <= -20.10838972759311
 CBF_Constraint_obs2: - 0.1238190319233109 x1_input_acceleration
   - 0.3119120679811496 x2_input_acceleration
   + 1.041999921133837 x3_input_acceleration >= -1.447980158077032
 CBF_Constraint_obs3: 0.2561809680766891 x1_input_acceleration
   - 0.3119120679811496 x2_input_acceleration
   + 1.041999921133837 x3_input_acceleration >= -1.510852077749887
 CBF_Constraint_obs1: - 0.4438190319233109 x1_input_acceleration
   - 0.3119120679811496 x2_input_acceleration
   + 1.041999921133837 x3_input_acceleration >= -1.675035383615681
 CBF_Constraint_for_table: x3_input_acceleration >= -2.844784436947904
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5218742586626217 x3_input_acceleration + relaxation_CLF
   <= -20.10894109674187
 CBF_Constraint_obs2: - 0.1236812259845902 x1_input_acceleration
   - 0.3115192733720797 x2_input_acceleration
   + 1.044948071563253 x3_input_acceleration >= -1.455322219508852
 CBF_Constraint_obs3: 0.2563187740154098 x1_input_acceleration
   - 0.3115192733720797 x2_input_acceleration
   + 1.044948071563253 x3_input_acceleration >= -1.518325054823492
 CBF_Constraint_obs1: - 0.4436812259845902 x1_input_acceleration
   - 0.3115192733720797 x2_input_acceleration
   + 1.044948071563253 x3_input_acceleration >= -1.682267200296524
 CBF_Constraint_for_table: x3_input_acceleration >= -2.851396045893454
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5231965804517317 x3_input_acceleration + relaxation_CLF
   <= -20.10949386471856
 CBF_Constraint_obs2: - 0.1237727787189652 x1_input_acceleration
   - 0.3115477643922824 x2_input_acceleration
   + 1.047483877060113 x3_input_acceleration >= -1.462005228688162
 CBF_Constraint_obs3: 0.2563187740154098 x1_input_acceleration
   - 0.3115192733720797 x2_input_acceleration
   + 1.047592715141473 x3_input_acceleration >= -1.525242585515644
 CBF_Constraint_obs1: - 0.4436812259845902 x1_input_acceleration
   - 0.3115192733720797 x2_input_acceleration
   + 1.047592715141473 x3_input_acceleration >= -1.689184730988677
 CBF_Constraint_for_table: x3_input_acceleration >= -2.857558533840621
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5245189022408416 x3_input_acceleration + relaxation_CLF
   <= -20.11004803152317
 CBF_Constraint_obs2: - 0.1237727787189652 x1_input_acceleration
   - 0.3115477643922824 x2_input_acceleration
   + 1.050128520638333 x3_input_acceleration >= -1.468939525134622
 CBF_Constraint_obs3: 0.2562272212810348 x1_input_acceleration
   - 0.3115477643922824 x2_input_acceleration
   + 1.050128520638333 x3_input_acceleration >= -1.531855385351605
 CBF_Constraint_obs1: - 0.4437727787189652 x1_input_acceleration
   - 0.3115477643922824 x2_input_acceleration
   + 1.050128520638333 x3_input_acceleration >= -1.695957748109794
 CBF_Constraint_for_table: x3_input_acceleration >= -2.864170142786171
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5258412240299516 x3_input_acceleration + relaxation_CLF
   <= -20.11060359715573
 CBF_Constraint_obs2: - 0.1234432842426468 x1_input_acceleration
   - 0.3114490591005343 x2_input_acceleration
   + 1.052888439599549 x3_input_acceleration >= -1.47601603385199
 CBF_Constraint_obs3: 0.2565567157573532 x1_input_acceleration
   - 0.3114490591005343 x2_input_acceleration
   + 1.052888439599549 x3_input_acceleration >= -1.539244913821475
 CBF_Constraint_obs1: - 0.4434432842426468 x1_input_acceleration
   - 0.3114490591005343 x2_input_acceleration
   + 1.052888439599549 x3_input_acceleration >= -1.702770661246107
 CBF_Constraint_for_table: x3_input_acceleration >= -2.870777579406587
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5271635458190616 x3_input_acceleration + relaxation_CLF
   <= -20.11116056161621
 CBF_Constraint_obs2: - 0.1237079288654496 x1_input_acceleration
   - 0.311501153560068 x2_input_acceleration
   + 1.055459650255405 x3_input_acceleration >= -1.482914617063369
 CBF_Constraint_obs3: 0.2562920711345504 x1_input_acceleration
   - 0.311501153560068 x2_input_acceleration
   + 1.055459650255405 x3_input_acceleration >= -1.545892084641192
 CBF_Constraint_obs1: - 0.4437079288654496 x1_input_acceleration
   - 0.311501153560068 x2_input_acceleration
   + 1.055459650255405 x3_input_acceleration >= -1.709880960155728
 CBF_Constraint_for_table: x3_input_acceleration >= -2.87786453539422
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5284858676081715 x3_input_acceleration + relaxation_CLF
   <= -20.11171892490463
 CBF_Constraint_obs2: - 0.1237804081134964 x1_input_acceleration
   - 0.3115339361146945 x2_input_acceleration
   + 1.058081405650032 x3_input_acceleration >= -1.489889054707756
 CBF_Constraint_obs3: 0.2562195918865036 x1_input_acceleration
   - 0.3115339361146945 x2_input_acceleration
   + 1.058081405650032 x3_input_acceleration >= -1.552797666999935
 CBF_Constraint_obs1: - 0.4437804081134964 x1_input_acceleration
   - 0.3115339361146945 x2_input_acceleration
   + 1.058081405650032 x3_input_acceleration >= -1.716913381198553
 CBF_Constraint_for_table: x3_input_acceleration >= -2.884031493689745
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5298081893972815 x3_input_acceleration + relaxation_CLF
   <= -20.11227868702097
 CBF_Constraint_obs2: - 0.1237804081134964 x1_input_acceleration
   - 0.3115339361146945 x2_input_acceleration
   + 1.060726049228252 x3_input_acceleration >= -1.496893417869042
 CBF_Constraint_obs3: 0.2562195918865036 x1_input_acceleration
   - 0.3115339361146945 x2_input_acceleration
   + 1.060726049228252 x3_input_acceleration >= -1.55980203016122
 CBF_Constraint_obs1: - 0.4437804081134964 x1_input_acceleration
   - 0.3115339361146945 x2_input_acceleration
   + 1.060726049228252 x3_input_acceleration >= -1.723917744359839
 CBF_Constraint_for_table: x3_input_acceleration >= -2.890643102635295
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5311305111863914 x3_input_acceleration + relaxation_CLF
   <= -20.11283984796525
 CBF_Constraint_obs2: - 0.1233636524372269 x1_input_acceleration
   - 0.3114132963136691 x2_input_acceleration
   + 1.063516843395461 x3_input_acceleration >= -1.504081135061804
 CBF_Constraint_obs3: 0.2566363475627731 x1_input_acceleration
   - 0.3114132963136691 x2_input_acceleration
   + 1.063516843395461 x3_input_acceleration >= -1.567385665246438
 CBF_Constraint_obs1: - 0.4433636524372269 x1_input_acceleration
   - 0.3114132963136691 x2_input_acceleration
   + 1.063516843395461 x3_input_acceleration >= -1.730772057011585
 CBF_Constraint_for_table: x3_input_acceleration >= -2.89727587122974
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5324528329755014 x3_input_acceleration + relaxation_CLF
   <= -20.11340240773745
 CBF_Constraint_obs2: - 0.1237923290424515 x1_input_acceleration
   - 0.3115476451829928 x2_input_acceleration
   + 1.065996143689074 x3_input_acceleration >= -1.510917817885999
 CBF_Constraint_obs3: 0.2562076709575485 x1_input_acceleration
   - 0.3115476451829928 x2_input_acceleration
   + 1.065996143689074 x3_input_acceleration >= -1.573815105295669
 CBF_Constraint_obs1: - 0.4437923290424515 x1_input_acceleration
   - 0.3115476451829928 x2_input_acceleration
   + 1.065996143689074 x3_input_acceleration >= -1.73795168111996
 CBF_Constraint_for_table: x3_input_acceleration >= -2.903865128433499
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5337751547646113 x3_input_acceleration + relaxation_CLF
   <= -20.11396636633759
 CBF_Constraint_obs2: - 0.1237618114643265 x1_input_acceleration
   - 0.3115322671846408 x2_input_acceleration
   + 1.068672377729025 x3_input_acceleration >= -1.518037487993926
 CBF_Constraint_obs3: 0.2562381885356735 x1_input_acceleration
   - 0.3115322671846408 x2_input_acceleration
   + 1.068672377729025 x3_input_acceleration >= -1.580963767102816
 CBF_Constraint_obs1: - 0.4437618114643265 x1_input_acceleration
   - 0.3115322671846408 x2_input_acceleration
   + 1.068672377729025 x3_input_acceleration >= -1.745046937165387
 CBF_Constraint_for_table: x3_input_acceleration >= -2.910515778421377
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5350974765537213 x3_input_acceleration + relaxation_CLF
   <= -20.11453172376567
 CBF_Constraint_obs2: - 0.1237618114643265 x1_input_acceleration
   - 0.3115322671846408 x2_input_acceleration
   + 1.071317021307244 x3_input_acceleration >= -1.525111874520951
 CBF_Constraint_obs3: 0.2562381885356735 x1_input_acceleration
   - 0.3115322671846408 x2_input_acceleration
   + 1.071317021307244 x3_input_acceleration >= -1.588038153629841
 CBF_Constraint_obs1: - 0.4437618114643265 x1_input_acceleration
   - 0.3115322671846408 x2_input_acceleration
   + 1.071317021307244 x3_input_acceleration >= -1.752121323692412
 CBF_Constraint_for_table: x3_input_acceleration >= -2.917127387366927
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5364197983428313 x3_input_acceleration + relaxation_CLF
   <= -20.11509848002166
 CBF_Constraint_obs2: - 0.123706498353975 x1_input_acceleration
   - 0.3114898286775607 x2_input_acceleration
   + 1.073982168883267 x3_input_acceleration >= -1.532208637729731
 CBF_Constraint_obs3: 0.256293501646025 x1_input_acceleration
   - 0.3114898286775607 x2_input_acceleration
   + 1.073982168883267 x3_input_acceleration >= -1.595187464293455
 CBF_Constraint_obs1: - 0.443706498353975 x1_input_acceleration
   - 0.3114898286775607 x2_input_acceleration
   + 1.073982168883267 x3_input_acceleration >= -1.759173836412911
 CBF_Constraint_for_table: x3_input_acceleration >= -2.924153248593666
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5377421201319412 x3_input_acceleration + relaxation_CLF
   <= -20.1156666351056
 CBF_Constraint_obs2: - 0.1237732555561234 x1_input_acceleration
   - 0.3118431650117892 x2_input_acceleration
   + 1.076349293235413 x3_input_acceleration >= -1.53886722425488
 CBF_Constraint_obs3: 0.2562267444438766 x1_input_acceleration
   - 0.3118431650117892 x2_input_acceleration
   + 1.076349293235413 x3_input_acceleration >= -1.601782631476563
 CBF_Constraint_obs1: - 0.4437732555561234 x1_input_acceleration
   - 0.3118431650117892 x2_input_acceleration
   + 1.076349293235413 x3_input_acceleration >= -1.765885828699779
 CBF_Constraint_for_table: x3_input_acceleration >= -2.930797938117066
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5390644419210512 x3_input_acceleration + relaxation_CLF
   <= -20.11623618901746
 CBF_Constraint_obs2: - 0.1237842228107621 x1_input_acceleration
   - 0.3118363700822848 x2_input_acceleration
   + 1.078998347557346 x3_input_acceleration >= -1.546002361943055
 CBF_Constraint_obs3: 0.2562267444438766 x1_input_acceleration
   - 0.3118431650117892 x2_input_acceleration
   + 1.078993936813633 x3_input_acceleration >= -1.608907774766824
 CBF_Constraint_obs1: - 0.4437732555561234 x1_input_acceleration
   - 0.3118431650117892 x2_input_acceleration
   + 1.078993936813633 x3_input_acceleration >= -1.77301097199004
 CBF_Constraint_for_table: x3_input_acceleration >= -2.937409547062615
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5403867637101611 x3_input_acceleration + relaxation_CLF
   <= -20.11680714175726
 CBF_Constraint_obs2: - 0.1237842228107621 x1_input_acceleration
   - 0.3118363700822848 x2_input_acceleration
   + 1.081642991135566 x3_input_acceleration >= -1.553145019744568
 CBF_Constraint_obs3: 0.2562157771892379 x1_input_acceleration
   - 0.3118363700822848 x2_input_acceleration
   + 1.081642991135566 x3_input_acceleration >= -1.616050008074344
 CBF_Constraint_obs1: - 0.4437842228107621 x1_input_acceleration
   - 0.3118363700822848 x2_input_acceleration
   + 1.081642991135566 x3_input_acceleration >= -1.780172397993177
 CBF_Constraint_for_table: x3_input_acceleration >= -2.943995526010912
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5417090854992711 x3_input_acceleration + relaxation_CLF
   <= -20.11737949332498
 CBF_Constraint_obs2: - 0.1236974384479691 x1_input_acceleration
   - 0.3114944778398532 x2_input_acceleration
   + 1.084547987802165 x3_input_acceleration >= -1.560717754811296
 CBF_Constraint_obs3: 0.2562157771892379 x1_input_acceleration
   - 0.3118363700822848 x2_input_acceleration
   + 1.084287634713786 x3_input_acceleration >= -1.623210151224997
 CBF_Constraint_obs1: - 0.4437842228107621 x1_input_acceleration
   - 0.3118363700822848 x2_input_acceleration
   + 1.084287634713786 x3_input_acceleration >= -1.78733254114383
 CBF_Constraint_for_table: x3_input_acceleration >= -2.950591935772044
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5430314072883811 x3_input_acceleration + relaxation_CLF
   <= -20.11795324372064
 CBF_Constraint_obs2: - 0.1237575199299027 x1_input_acceleration
   - 0.3115788780168551 x2_input_acceleration
   + 1.087214565889662 x3_input_acceleration >= -1.568041041560111
 CBF_Constraint_obs3: 0.2562424800700973 x1_input_acceleration
   - 0.3115788780168551 x2_input_acceleration
   + 1.087214565889662 x3_input_acceleration >= -1.630971397626704
 CBF_Constraint_obs1: - 0.4437575199299027 x1_input_acceleration
   - 0.3115788780168551 x2_input_acceleration
   + 1.087214565889662 x3_input_acceleration >= -1.795047057504033
 CBF_Constraint_for_table: x3_input_acceleration >= -2.957213081460758
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.544353729077491 x3_input_acceleration + relaxation_CLF
   <= -20.11852839294423
 CBF_Constraint_obs2: - 0.1237575199299027 x1_input_acceleration
   - 0.3115788780168551 x2_input_acceleration
   + 1.089859209467882 x3_input_acceleration >= -1.575238021784249
 CBF_Constraint_obs3: 0.2562424800700973 x1_input_acceleration
   - 0.3115788780168551 x2_input_acceleration
   + 1.089859209467882 x3_input_acceleration >= -1.638168377850842
 CBF_Constraint_obs1: - 0.4437575199299027 x1_input_acceleration
   - 0.3115788780168551 x2_input_acceleration
   + 1.089859209467882 x3_input_acceleration >= -1.802244037728171
 CBF_Constraint_for_table: x3_input_acceleration >= -2.963824690406307
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.545676050866601 x3_input_acceleration + relaxation_CLF
   <= -20.11910494099575
 CBF_Constraint_obs2: - 0.1238752987079789 x1_input_acceleration
   - 0.3118833385423678 x2_input_acceleration
   + 1.092158503734273 x3_input_acceleration >= -1.581783129711764
 CBF_Constraint_obs3: 0.2561247012920211 x1_input_acceleration
   - 0.3118833385423678 x2_input_acceleration
   + 1.092158503734273 x3_input_acceleration >= -1.644601595939184
 CBF_Constraint_obs1: - 0.4438752987079789 x1_input_acceleration
   - 0.3118833385423678 x2_input_acceleration
   + 1.092158503734273 x3_input_acceleration >= -1.808883368678147
 CBF_Constraint_for_table: x3_input_acceleration >= -2.970443153886007
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5469983726557109 x3_input_acceleration + relaxation_CLF
   <= -20.1196828878752
 CBF_Constraint_obs2: - 0.1238752987079789 x1_input_acceleration
   - 0.3118833385423678 x2_input_acceleration
   + 1.094803147312493 x3_input_acceleration >= -1.589012797319581
 CBF_Constraint_obs3: 0.2561247012920211 x1_input_acceleration
   - 0.3118833385423678 x2_input_acceleration
   + 1.094803147312493 x3_input_acceleration >= -1.651831263547001
 CBF_Constraint_obs1: - 0.4438752987079789 x1_input_acceleration
   - 0.3118833385423678 x2_input_acceleration
   + 1.094803147312493 x3_input_acceleration >= -1.816113036285965
 CBF_Constraint_for_table: x3_input_acceleration >= -2.977054762831556
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5483206944448209 x3_input_acceleration + relaxation_CLF
   <= -20.12026223358258
 CBF_Constraint_obs2: - 0.1234480526142289 x1_input_acceleration
   - 0.3112989746049899 x2_input_acceleration
   + 1.098952092915555 x3_input_acceleration >= -1.599802719759801
 CBF_Constraint_obs3: 0.2561247012920211 x1_input_acceleration
   - 0.3118833385423678 x2_input_acceleration
   + 1.097447790890713 x3_input_acceleration >= -1.659078416503958
 CBF_Constraint_obs1: - 0.4438752987079789 x1_input_acceleration
   - 0.3118833385423678 x2_input_acceleration
   + 1.097447790890713 x3_input_acceleration >= -1.823360189242921
 CBF_Constraint_for_table: x3_input_acceleration >= -2.983234536125709
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5496430162339309 x3_input_acceleration + relaxation_CLF
   <= -20.12084297811789
 CBF_Constraint_obs2: - 0.1234480526142289 x1_input_acceleration
   - 0.3112989746049899 x2_input_acceleration
   + 1.101596736493774 x3_input_acceleration >= -1.607077303922622
 CBF_Constraint_obs3: 0.2565519473857711 x1_input_acceleration
   - 0.3112989746049899 x2_input_acceleration
   + 1.101596736493774 x3_input_acceleration >= -1.670301653939105
 CBF_Constraint_obs1: - 0.4438752987079789 x1_input_acceleration
   - 0.3118833385423678 x2_input_acceleration
   + 1.100092434468933 x3_input_acceleration >= -1.830624827549018
 CBF_Constraint_for_table: x3_input_acceleration >= -2.989793692983856
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5509653380230408 x3_input_acceleration + relaxation_CLF
   <= -20.12142512148114
 CBF_Constraint_obs2: - 0.1237932827167679 x1_input_acceleration
   - 0.3115414462999362 x2_input_acceleration
   + 1.103017100668308 x3_input_acceleration >= -1.611286967468962
 CBF_Constraint_obs3: 0.2562067172832321 x1_input_acceleration
   - 0.3115414462999362 x2_input_acceleration
   + 1.103017100668308 x3_input_acceleration >= -1.674183348888032
 CBF_Constraint_obs1: - 0.4437932827167679 x1_input_acceleration
   - 0.3115414462999362 x2_input_acceleration
   + 1.103017100668308 x3_input_acceleration >= -1.838321593642376
 CBF_Constraint_for_table: x3_input_acceleration >= -2.996423779369286
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5522876598121508 x3_input_acceleration + relaxation_CLF
   <= -20.12200866367231
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.311529286952402 x2_input_acceleration
   + 1.105662221083686 x3_input_acceleration >= -1.618574520839893
 CBF_Constraint_obs3: 0.2562067172832321 x1_input_acceleration
   - 0.3115414462999362 x2_input_acceleration
   + 1.105661744246528 x3_input_acceleration >= -1.681484809292475
 CBF_Constraint_obs1: - 0.4437932827167679 x1_input_acceleration
   - 0.3115414462999362 x2_input_acceleration
   + 1.105661744246528 x3_input_acceleration >= -1.845623054046819
 CBF_Constraint_for_table: x3_input_acceleration >= -3.003105125749223
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5536099816012607 x3_input_acceleration + relaxation_CLF
   <= -20.12259360469142
 CBF_Constraint_obs2: - 0.1237732555561234 x1_input_acceleration
   - 0.311864980311777 x2_input_acceleration
   + 1.108046153945658 x3_input_acceleration >= -1.62543233001199
 CBF_Constraint_obs3: 0.2562267444438766 x1_input_acceleration
   - 0.311864980311777 x2_input_acceleration
   + 1.108046153945658 x3_input_acceleration >= -1.688347737233673
 CBF_Constraint_obs1: - 0.4437732555561234 x1_input_acceleration
   - 0.311864980311777 x2_input_acceleration
   + 1.108046153945658 x3_input_acceleration >= -1.852450934456889
 CBF_Constraint_for_table: x3_input_acceleration >= -3.009658620166117
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5549323033903707 x3_input_acceleration + relaxation_CLF
   <= -20.12317994453846
 CBF_Constraint_obs2: - 0.1237732555561234 x1_input_acceleration
   - 0.311864980311777 x2_input_acceleration
   + 1.110690797523878 x3_input_acceleration >= -1.632767040550069
 CBF_Constraint_obs3: 0.2562267444438766 x1_input_acceleration
   - 0.311864980311777 x2_input_acceleration
   + 1.110690797523878 x3_input_acceleration >= -1.695682447771752
 CBF_Constraint_obs1: - 0.4437732555561234 x1_input_acceleration
   - 0.311864980311777 x2_input_acceleration
   + 1.110690797523878 x3_input_acceleration >= -1.859785644994968
 CBF_Constraint_for_table: x3_input_acceleration >= -3.016270229111667
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5562546251794807 x3_input_acceleration + relaxation_CLF
   <= -20.12376768321343
 CBF_Constraint_obs2: - 0.1238671924762894 x1_input_acceleration
   - 0.3118786893800753 x2_input_acceleration
   + 1.113334249009203 x3_input_acceleration >= -1.640155685347846
 CBF_Constraint_obs3: 0.2562267444438766 x1_input_acceleration
   - 0.311864980311777 x2_input_acceleration
   + 1.113335441102098 x3_input_acceleration >= -1.70303464365897
 CBF_Constraint_obs1: - 0.4437732555561234 x1_input_acceleration
   - 0.311864980311777 x2_input_acceleration
   + 1.113335441102098 x3_input_acceleration >= -1.867137840882186
 CBF_Constraint_for_table: x3_input_acceleration >= -3.022930713865932
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5575769469685906 x3_input_acceleration + relaxation_CLF
   <= -20.12435682071633
 CBF_Constraint_obs2: - 0.1238385822467972 x1_input_acceleration
   - 0.3118951402620334 x2_input_acceleration
   + 1.115972216867208 x3_input_acceleration >= -1.64751070223935
 CBF_Constraint_obs3: 0.2561328075237106 x1_input_acceleration
   - 0.3118786893800753 x2_input_acceleration
   + 1.115978892587423 x3_input_acceleration >= -1.710351525850077
 CBF_Constraint_obs1: - 0.4438671924762894 x1_input_acceleration
   - 0.3118786893800753 x2_input_acceleration
   + 1.115978892587423 x3_input_acceleration >= -1.874619112683584
 CBF_Constraint_for_table: x3_input_acceleration >= -3.029538746532796
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5588992687577006 x3_input_acceleration + relaxation_CLF
   <= -20.12494735704716
 CBF_Constraint_obs2: - 0.1237742092304398 x1_input_acceleration
   - 0.3115272603944796 x2_input_acceleration
   + 1.118908923204827 x3_input_acceleration >= -1.65540808408362
 CBF_Constraint_obs3: 0.2562257907695602 x1_input_acceleration
   - 0.3115272603944796 x2_input_acceleration
   + 1.118908923204827 x3_input_acceleration >= -1.718322585314702
 CBF_Constraint_obs1: - 0.4437742092304398 x1_input_acceleration
   - 0.3115272603944796 x2_input_acceleration
   + 1.118908923204827 x3_input_acceleration >= -1.882427451467972
 CBF_Constraint_for_table: x3_input_acceleration >= -3.036102671762525
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5602215905468105 x3_input_acceleration + relaxation_CLF
   <= -20.12553929220592
 CBF_Constraint_obs2: - 0.1238514568500687 x1_input_acceleration
   - 0.3118897758440036 x2_input_acceleration
   + 1.121257808535671 x3_input_acceleration >= -1.662291861052655
 CBF_Constraint_obs3: 0.2561485431499313 x1_input_acceleration
   - 0.3118897758440036 x2_input_acceleration
   + 1.121257808535671 x3_input_acceleration >= -1.72513297704509
 CBF_Constraint_obs1: - 0.4438514568500687 x1_input_acceleration
   - 0.3118897758440036 x2_input_acceleration
   + 1.121257808535671 x3_input_acceleration >= -1.88937302653271
 CBF_Constraint_for_table: x3_input_acceleration >= -3.042750341518164
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5615439123359205 x3_input_acceleration + relaxation_CLF
   <= -20.12613262619261
 CBF_Constraint_obs2: - 0.1238514568500687 x1_input_acceleration
   - 0.3118897758440036 x2_input_acceleration
   + 1.123902452113891 x3_input_acceleration >= -1.669713921884406
 CBF_Constraint_obs3: 0.2561485431499313 x1_input_acceleration
   - 0.3118897758440036 x2_input_acceleration
   + 1.123902452113891 x3_input_acceleration >= -1.732555037876841
 CBF_Constraint_obs1: - 0.4438514568500687 x1_input_acceleration
   - 0.3118897758440036 x2_input_acceleration
   + 1.123902452113891 x3_input_acceleration >= -1.896795087364461
 CBF_Constraint_for_table: x3_input_acceleration >= -3.049361950463714
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5628662341250305 x3_input_acceleration + relaxation_CLF
   <= -20.12672735900724
 CBF_Constraint_obs2: - 0.1238738681965043 x1_input_acceleration
   - 0.3118720136598605 x2_input_acceleration
   + 1.12654733411069 x3_input_acceleration >= -1.677147230147656
 CBF_Constraint_obs3: 0.2561261318034957 x1_input_acceleration
   - 0.3118720136598605 x2_input_acceleration
   + 1.12654733411069 x3_input_acceleration >= -1.739967055360977
 CBF_Constraint_obs1: - 0.4438514568500687 x1_input_acceleration
   - 0.3118897758440036 x2_input_acceleration
   + 1.126547095692111 x3_input_acceleration >= -1.904234633545353
 CBF_Constraint_for_table: x3_input_acceleration >= -3.055970877200249
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5641885559141404 x3_input_acceleration + relaxation_CLF
   <= -20.12732349064979
 CBF_Constraint_obs2: - 0.1238705303363968 x1_input_acceleration
   - 0.311889656634714 x2_input_acceleration
   + 1.129190785596015 x3_input_acceleration >= -1.684613620595606
 CBF_Constraint_obs3: 0.2561294696636032 x1_input_acceleration
   - 0.311889656634714 x2_input_acceleration
   + 1.129190785596015 x3_input_acceleration >= -1.747436616776029
 CBF_Constraint_obs1: - 0.4438705303363968 x1_input_acceleration
   - 0.311889656634714 x2_input_acceleration
   + 1.129190785596015 x3_input_acceleration >= -1.911710044864724
 CBF_Constraint_for_table: x3_input_acceleration >= -3.062556558125321
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5655108777032504 x3_input_acceleration + relaxation_CLF
   <= -20.12792102112028
 CBF_Constraint_obs2: - 0.1237022068195511 x1_input_acceleration
   - 0.3114892326311129 x2_input_acceleration
   + 1.132154194814494 x3_input_acceleration >= -1.692626123566501
 CBF_Constraint_obs3: 0.2562977931804489 x1_input_acceleration
   - 0.3114892326311129 x2_input_acceleration
   + 1.132154194814494 x3_input_acceleration >= -1.755609027087927
 CBF_Constraint_obs1: - 0.4437022068195511 x1_input_acceleration
   - 0.3114892326311129 x2_input_acceleration
   + 1.132154194814494 x3_input_acceleration >= -1.919587889022142
 CBF_Constraint_for_table: x3_input_acceleration >= -3.069176809744363
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5668331994923603 x3_input_acceleration + relaxation_CLF
   <= -20.1285199504187
 CBF_Constraint_obs2: - 0.1237022068195511 x1_input_acceleration
   - 0.3114892326311129 x2_input_acceleration
   + 1.134798838392713 x3_input_acceleration >= -1.700120227043248
 CBF_Constraint_obs3: 0.2562977931804489 x1_input_acceleration
   - 0.3114892326311129 x2_input_acceleration
   + 1.134798838392713 x3_input_acceleration >= -1.763103130564674
 CBF_Constraint_obs1: - 0.4437022068195511 x1_input_acceleration
   - 0.3114892326311129 x2_input_acceleration
   + 1.134798838392713 x3_input_acceleration >= -1.927081992498889
 CBF_Constraint_for_table: x3_input_acceleration >= -3.075788418689913
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5681555212814703 x3_input_acceleration + relaxation_CLF
   <= -20.12912027854505
 CBF_Constraint_obs2: - 0.1238423969440628 x1_input_acceleration
   - 0.311863907428171 x2_input_acceleration
   + 1.137173949767259 x3_input_acceleration >= -1.707200785083108
 CBF_Constraint_obs3: 0.2561576030559372 x1_input_acceleration
   - 0.311863907428171 x2_input_acceleration
   + 1.137173949767259 x3_input_acceleration >= -1.770050507986248
 CBF_Constraint_obs1: - 0.4437022068195511 x1_input_acceleration
   - 0.3114892326311129 x2_input_acceleration
   + 1.137443481970933 x3_input_acceleration >= -1.934593581324775
 CBF_Constraint_for_table: x3_input_acceleration >= -3.08284795654095
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5694778430705802 x3_input_acceleration + relaxation_CLF
   <= -20.12972200549933
 CBF_Constraint_obs2: - 0.1237837459736039 x1_input_acceleration
   - 0.3118342243150729 x2_input_acceleration
   + 1.139827653251485 x3_input_acceleration >= -1.714712597922785
 CBF_Constraint_obs3: 0.2562162540263961 x1_input_acceleration
   - 0.3118342243150729 x2_input_acceleration
   + 1.139827653251485 x3_input_acceleration >= -1.777618039247861
 CBF_Constraint_obs1: - 0.4437837459736039 x1_input_acceleration
   - 0.3118342243150729 x2_input_acceleration
   + 1.139827653251485 x3_input_acceleration >= -1.941739594701668
 CBF_Constraint_for_table: x3_input_acceleration >= -3.089027431811878
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5708001648596902 x3_input_acceleration + relaxation_CLF
   <= -20.13032513128154
 CBF_Constraint_obs2: - 0.1237870838337113 x1_input_acceleration
   - 0.3115371547655124 x2_input_acceleration
   + 1.142682820435051 x3_input_acceleration >= -1.722628336301061
 CBF_Constraint_obs3: 0.2562129161662887 x1_input_acceleration
   - 0.3115371547655124 x2_input_acceleration
   + 1.142682820435051 x3_input_acceleration >= -1.785530606659036
 CBF_Constraint_obs1: - 0.4437837459736039 x1_input_acceleration
   - 0.3118342243150729 x2_input_acceleration
   + 1.142472296829705 x3_input_acceleration >= -1.94928443208486
 CBF_Constraint_for_table: x3_input_acceleration >= -3.096043160249005
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5721224866488002 x3_input_acceleration + relaxation_CLF
   <= -20.13092965589168
 CBF_Constraint_obs2: - 0.1237870838337113 x1_input_acceleration
   - 0.3115371547655124 x2_input_acceleration
   + 1.145327464013271 x3_input_acceleration >= -1.730192050933146
 CBF_Constraint_obs3: 0.2562129161662887 x1_input_acceleration
   - 0.3115371547655124 x2_input_acceleration
   + 1.145327464013271 x3_input_acceleration >= -1.79309432129112
 CBF_Constraint_obs1: - 0.4437870838337113 x1_input_acceleration
   - 0.3115371547655124 x2_input_acceleration
   + 1.145327464013271 x3_input_acceleration >= -1.957221718000115
 CBF_Constraint_for_table: x3_input_acceleration >= -3.102654769194555
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5734448084379101 x3_input_acceleration + relaxation_CLF
   <= -20.13153557932976
 CBF_Constraint_obs2: - 0.1237751629047562 x1_input_acceleration
   - 0.3115224920228976 x2_input_acceleration
   + 1.147999406518798 x3_input_acceleration >= -1.737836489194711
 CBF_Constraint_obs3: 0.2562248370952438 x1_input_acceleration
   - 0.3115224920228976 x2_input_acceleration
   + 1.147999406518798 x3_input_acceleration >= -1.800750084435192
 CBF_Constraint_obs1: - 0.4437751629047562 x1_input_acceleration
   - 0.3115224920228976 x2_input_acceleration
   + 1.147999406518798 x3_input_acceleration >= -1.964856619518516
 CBF_Constraint_for_table: x3_input_acceleration >= -3.109307803368224
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5747671302270201 x3_input_acceleration + relaxation_CLF
   <= -20.13214290159576
 CBF_Constraint_obs2: - 0.1237780239277054 x1_input_acceleration
   - 0.3115332208589572 x2_input_acceleration
   + 1.150619969820529 x3_input_acceleration >= -1.745375327319998
 CBF_Constraint_obs3: 0.2562219760722946 x1_input_acceleration
   - 0.3115332208589572 x2_input_acceleration
   + 1.150619969820529 x3_input_acceleration >= -1.808286204588678
 CBF_Constraint_obs1: - 0.4437751629047562 x1_input_acceleration
   - 0.3115224920228976 x2_input_acceleration
   + 1.150644050097018 x3_input_acceleration >= -1.972455485338711
 CBF_Constraint_for_table: x3_input_acceleration >= -3.115827919184043
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.57608945201613 x3_input_acceleration + relaxation_CLF
   <= -20.1327516226897
 CBF_Constraint_obs2: - 0.1237780239277054 x1_input_acceleration
   - 0.3115332208589572 x2_input_acceleration
   + 1.153264613398749 x3_input_acceleration >= -1.752991519279962
 CBF_Constraint_obs3: 0.2562219760722946 x1_input_acceleration
   - 0.3115332208589572 x2_input_acceleration
   + 1.153264613398749 x3_input_acceleration >= -1.815902396548641
 CBF_Constraint_obs1: - 0.4437780239277054 x1_input_acceleration
   - 0.3115332208589572 x2_input_acceleration
   + 1.153264613398749 x3_input_acceleration >= -1.980013938422126
 CBF_Constraint_for_table: x3_input_acceleration >= -3.122488105915085
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.57741177380524 x3_input_acceleration + relaxation_CLF
   <= -20.13336174261157
 CBF_Constraint_obs2: - 0.1237918522052933 x1_input_acceleration
   - 0.3115413270906466 x2_input_acceleration
   + 1.155929641765482 x3_input_acceleration >= -1.760694697337367
 CBF_Constraint_obs3: 0.2562081477947067 x1_input_acceleration
   - 0.3115413270906466 x2_input_acceleration
   + 1.155929641765482 x3_input_acceleration >= -1.823592437742339
 CBF_Constraint_obs1: - 0.443871007173555 x1_input_acceleration
   - 0.3118797622636813 x2_input_acceleration
   + 1.155636267703898 x3_input_acceleration >= -1.987232050439389
 CBF_Constraint_for_table: x3_input_acceleration >= -3.129120576486306
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.57873409559435 x3_input_acceleration + relaxation_CLF
   <= -20.13397326136137
 CBF_Constraint_obs2: - 0.1238719608478714 x1_input_acceleration
   - 0.3118761859849948 x2_input_acceleration
   + 1.158300819233473 x3_input_acceleration >= -1.767839757382221
 CBF_Constraint_obs3: 0.2561280391521286 x1_input_acceleration
   - 0.3118761859849948 x2_input_acceleration
   + 1.158300819233473 x3_input_acceleration >= -1.830661394576744
 CBF_Constraint_obs1: - 0.4438719608478714 x1_input_acceleration
   - 0.3118761859849948 x2_input_acceleration
   + 1.158300819233473 x3_input_acceleration >= -1.994937326060519
 CBF_Constraint_for_table: x3_input_acceleration >= -3.135310780593294
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5800564173834599 x3_input_acceleration + relaxation_CLF
   <= -20.1345861789391
 CBF_Constraint_obs2: - 0.1238719608478714 x1_input_acceleration
   - 0.3118761859849948 x2_input_acceleration
   + 1.160945462811692 x3_input_acceleration >= -1.775506732114873
 CBF_Constraint_obs3: 0.2561280391521286 x1_input_acceleration
   - 0.3118761859849948 x2_input_acceleration
   + 1.160945462811692 x3_input_acceleration >= -1.838328369309395
 CBF_Constraint_obs1: - 0.4438719608478714 x1_input_acceleration
   - 0.3118761859849948 x2_input_acceleration
   + 1.160945462811692 x3_input_acceleration >= -2.00260430079317
 CBF_Constraint_for_table: x3_input_acceleration >= -3.141922389538844
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5813787391725699 x3_input_acceleration + relaxation_CLF
   <= -20.13520049534475
 CBF_Constraint_obs2: - 0.123685517519014 x1_input_acceleration
   - 0.3114988885835666 x2_input_acceleration
   + 1.163905176542195 x3_input_acceleration >= -1.783756156018077
 CBF_Constraint_obs3: 0.256314482480986 x1_input_acceleration
   - 0.3114988885835666 x2_input_acceleration
   + 1.163905176542195 x3_input_acceleration >= -1.846754914375014
 CBF_Constraint_obs1: - 0.443685517519014 x1_input_acceleration
   - 0.3114988885835666 x2_input_acceleration
   + 1.163905176542195 x3_input_acceleration >= -2.010704570033289
 CBF_Constraint_for_table: x3_input_acceleration >= -3.14854144906499
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5827010609616798 x3_input_acceleration + relaxation_CLF
   <= -20.13581621057835
 CBF_Constraint_obs2: - 0.123685517519014 x1_input_acceleration
   - 0.3114988885835666 x2_input_acceleration
   + 1.166549820120415 x3_input_acceleration >= -1.791460184569645
 CBF_Constraint_obs3: 0.256314482480986 x1_input_acceleration
   - 0.3114988885835666 x2_input_acceleration
   + 1.166549820120415 x3_input_acceleration >= -1.854458942926582
 CBF_Constraint_obs1: - 0.443685517519014 x1_input_acceleration
   - 0.3114988885835666 x2_input_acceleration
   + 1.166549820120415 x3_input_acceleration >= -2.018408598584856
 CBF_Constraint_for_table: x3_input_acceleration >= -3.15515305801054
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5840233827507898 x3_input_acceleration + relaxation_CLF
   <= -20.13643332463987
 CBF_Constraint_obs2: - 0.1238490726642777 x1_input_acceleration
   - 0.3119363866762179 x2_input_acceleration
   + 1.168891552893886 x3_input_acceleration >= -1.798687955698872
 CBF_Constraint_obs3: 0.2562396190471481 x1_input_acceleration
   - 0.3118612848238009 x2_input_acceleration
   + 1.16888261219717 x3_input_acceleration >= -1.861503455547824
 CBF_Constraint_obs1: - 0.4437603809528519 x1_input_acceleration
   - 0.3118612848238009 x2_input_acceleration
   + 1.16888261219717 x3_input_acceleration >= -2.025584122215315
 CBF_Constraint_for_table: x3_input_acceleration >= -3.16180460206809
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5853457045398998 x3_input_acceleration + relaxation_CLF
   <= -20.13705183752932
 CBF_Constraint_obs2: - 0.1238471653156449 x1_input_acceleration
   - 0.3118738017992038 x2_input_acceleration
   + 1.171544898750243 x3_input_acceleration >= -1.80640104799141
 CBF_Constraint_obs3: 0.2561509273357223 x1_input_acceleration
   - 0.3119363866762179 x2_input_acceleration
   + 1.171536196472106 x3_input_acceleration >= -1.869268333189869
 CBF_Constraint_obs1: - 0.4438490726642777 x1_input_acceleration
   - 0.3119363866762179 x2_input_acceleration
   + 1.171536196472106 x3_input_acceleration >= -2.033504210352355
 CBF_Constraint_for_table: x3_input_acceleration >= -3.168810793762052
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5866680263290097 x3_input_acceleration + relaxation_CLF
   <= -20.13767174924671
 CBF_Constraint_obs2: - 0.1238471653156449 x1_input_acceleration
   - 0.3118738017992038 x2_input_acceleration
   + 1.174189542328463 x3_input_acceleration >= -1.81415558739867
 CBF_Constraint_obs3: 0.2561528346843551 x1_input_acceleration
   - 0.3118738017992038 x2_input_acceleration
   + 1.174189542328463 x3_input_acceleration >= -1.877000780348808
 CBF_Constraint_obs1: - 0.4438471653156449 x1_input_acceleration
   - 0.3118738017992038 x2_input_acceleration
   + 1.174189542328463 x3_input_acceleration >= -2.041233319651186
 CBF_Constraint_for_table: x3_input_acceleration >= -3.175422402707602
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5879903481181197 x3_input_acceleration + relaxation_CLF
   <= -20.13829305979203
 CBF_Constraint_obs2: - 0.1237737323932816 x1_input_acceleration
   - 0.3118988357500094 x2_input_acceleration
   + 1.176852305718695 x3_input_acceleration >= -1.82197771256023
 CBF_Constraint_obs3: 0.2562262676067184 x1_input_acceleration
   - 0.3118988357500094 x2_input_acceleration
   + 1.176852305718695 x3_input_acceleration >= -1.884892666786612
 CBF_Constraint_obs1: - 0.4438471653156449 x1_input_acceleration
   - 0.3118738017992038 x2_input_acceleration
   + 1.176834185906683 x3_input_acceleration >= -2.049005344407586
 CBF_Constraint_for_table: x3_input_acceleration >= -3.181561644843307
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5893126699072296 x3_input_acceleration + relaxation_CLF
   <= -20.13891576916528
 CBF_Constraint_obs2: - 0.1237789776020218 x1_input_acceleration
   - 0.3115396581605929 x2_input_acceleration
   + 1.179727500062906 x3_input_acceleration >= -1.830168960384619
 CBF_Constraint_obs3: 0.2562210223979782 x1_input_acceleration
   - 0.3115396581605929 x2_input_acceleration
   + 1.179727500062906 x3_input_acceleration >= -1.893078931662698
 CBF_Constraint_obs1: - 0.4437789776020218 x1_input_acceleration
   - 0.3115396581605929 x2_input_acceleration
   + 1.179727500062906 x3_input_acceleration >= -2.057192142466237
 CBF_Constraint_for_table: x3_input_acceleration >= -3.188177426113991
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5906349916963396 x3_input_acceleration + relaxation_CLF
   <= -20.13953987736645
 CBF_Constraint_obs2: - 0.1237470295124222 x1_input_acceleration
   - 0.311893113704111 x2_input_acceleration
   + 1.182168414965284 x3_input_acceleration >= -1.837641003788406
 CBF_Constraint_obs3: 0.2562210223979782 x1_input_acceleration
   - 0.3115396581605929 x2_input_acceleration
   + 1.182372143641126 x3_input_acceleration >= -1.900887571229995
 CBF_Constraint_obs1: - 0.4437789776020218 x1_input_acceleration
   - 0.3115396581605929 x2_input_acceleration
   + 1.182372143641126 x3_input_acceleration >= -2.065000782033533
 CBF_Constraint_for_table: x3_input_acceleration >= -3.195186896063417
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5919573134854496 x3_input_acceleration + relaxation_CLF
   <= -20.14016538439557
 CBF_Constraint_obs2: - 0.1237470295124222 x1_input_acceleration
   - 0.311893113704111 x2_input_acceleration
   + 1.184813058543504 x3_input_acceleration >= -1.845465781730507
 CBF_Constraint_obs3: 0.2562529704875778 x1_input_acceleration
   - 0.311893113704111 x2_input_acceleration
   + 1.184813058543504 x3_input_acceleration >= -1.908406103693706
 CBF_Constraint_obs1: - 0.4437470295124222 x1_input_acceleration
   - 0.311893113704111 x2_input_acceleration
   + 1.184813058543504 x3_input_acceleration >= -2.072463405340445
 CBF_Constraint_for_table: x3_input_acceleration >= -3.201798505008966
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5932796352745595 x3_input_acceleration + relaxation_CLF
   <= -20.14079229025261
 CBF_Constraint_obs2: - 0.1237751629047562 x1_input_acceleration
   - 0.3115384660676974 x2_input_acceleration
   + 1.187669656238545 x3_input_acceleration >= -1.85366964892712
 CBF_Constraint_obs3: 0.2562248370952438 x1_input_acceleration
   - 0.3115384660676974 x2_input_acceleration
   + 1.187669656238545 x3_input_acceleration >= -1.916583244167602
 CBF_Constraint_obs1: - 0.4437751629047562 x1_input_acceleration
   - 0.3115384660676974 x2_input_acceleration
   + 1.187669656238545 x3_input_acceleration >= -2.080689779250925
 CBF_Constraint_for_table: x3_input_acceleration >= -3.208036988878222
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5946019570636695 x3_input_acceleration + relaxation_CLF
   <= -20.14142059493758
 CBF_Constraint_obs2: - 0.1238772060566117 x1_input_acceleration
   - 0.3118792854265231 x2_input_acceleration
   + 1.190019852871574 x3_input_acceleration >= -1.860951876358793
 CBF_Constraint_obs3: 0.2561227939433883 x1_input_acceleration
   - 0.3118792854265231 x2_input_acceleration
   + 1.190019852871574 x3_input_acceleration >= -1.923768530605012
 CBF_Constraint_obs1: - 0.4438772060566117 x1_input_acceleration
   - 0.3118792854265231 x2_input_acceleration
   + 1.190019852871574 x3_input_acceleration >= -2.088053641204082
 CBF_Constraint_for_table: x3_input_acceleration >= -3.214702838050518
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5959242788527794 x3_input_acceleration + relaxation_CLF
   <= -20.14205029845048
 CBF_Constraint_obs2: - 0.1238428737812211 x1_input_acceleration
   - 0.311931856723215 x2_input_acceleration
   + 1.192683331517543 x3_input_acceleration >= -1.868915087381212
 CBF_Constraint_obs3: 0.256157126218779 x1_input_acceleration
   - 0.311931856723215 x2_input_acceleration
   + 1.192683331517543 x3_input_acceleration >= -1.931764357289052
 CBF_Constraint_obs1: - 0.4438428737812211 x1_input_acceleration
   - 0.311931856723215 x2_input_acceleration
   + 1.192683331517543 x3_input_acceleration >= -2.095989386406189
 CBF_Constraint_for_table: x3_input_acceleration >= -3.221217589448307
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5972466006418894 x3_input_acceleration + relaxation_CLF
   <= -20.14268140079132
 CBF_Constraint_obs2: - 0.1238633777790238 x1_input_acceleration
   - 0.3118972860292453 x2_input_acceleration
   + 1.195285298170104 x3_input_acceleration >= -1.876661246761041
 CBF_Constraint_obs3: 0.256157126218779 x1_input_acceleration
   - 0.311931856723215 x2_input_acceleration
   + 1.195327975095763 x3_input_acceleration >= -1.939658655747492
 CBF_Constraint_obs1: - 0.4438428737812211 x1_input_acceleration
   - 0.311931856723215 x2_input_acceleration
   + 1.195327975095763 x3_input_acceleration >= -2.103883684864629
 CBF_Constraint_for_table: x3_input_acceleration >= -3.22786555722717
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5985689224309994 x3_input_acceleration + relaxation_CLF
   <= -20.14331390196008
 CBF_Constraint_obs2: - 0.1238633777790238 x1_input_acceleration
   - 0.3118972860292453 x2_input_acceleration
   + 1.197929941748324 x3_input_acceleration >= -1.884572748405476
 CBF_Constraint_obs3: 0.2561366222209762 x1_input_acceleration
   - 0.3118972860292453 x2_input_acceleration
   + 1.197929941748324 x3_input_acceleration >= -1.947402539515404
 CBF_Constraint_obs1: - 0.4438633777790238 x1_input_acceleration
   - 0.3118972860292453 x2_input_acceleration
   + 1.197929941748324 x3_input_acceleration >= -2.111663450628695
 CBF_Constraint_for_table: x3_input_acceleration >= -3.234435144898153
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.5998912442201093 x3_input_acceleration + relaxation_CLF
   <= -20.14394780195678
 CBF_Constraint_obs2: - 0.123866238801973 x1_input_acceleration
   - 0.3118845306352633 x2_input_acceleration
   + 1.200607725509039 x3_input_acceleration >= -1.892592145145643
 CBF_Constraint_obs3: 0.256133761198027 x1_input_acceleration
   - 0.3118845306352633 x2_input_acceleration
   + 1.200607725509039 x3_input_acceleration >= -1.955419218283768
 CBF_Constraint_obs1: - 0.4438633777790238 x1_input_acceleration
   - 0.3118972860292453 x2_input_acceleration
   + 1.200574585326544 x3_input_acceleration >= -2.11959243762227
 CBF_Constraint_for_table: x3_input_acceleration >= -3.241046753843703
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6012135660092193 x3_input_acceleration + relaxation_CLF
   <= -20.14458310078141
 CBF_Constraint_obs2: - 0.123866238801973 x1_input_acceleration
   - 0.3118845306352633 x2_input_acceleration
   + 1.203252369087259 x3_input_acceleration >= -1.900538836598284
 CBF_Constraint_obs3: 0.256133761198027 x1_input_acceleration
   - 0.3118845306352633 x2_input_acceleration
   + 1.203252369087259 x3_input_acceleration >= -1.96336590973641
 CBF_Constraint_obs1: - 0.443866238801973 x1_input_acceleration
   - 0.3118845306352633 x2_input_acceleration
   + 1.203252369087259 x3_input_acceleration >= -2.127631827639862
 CBF_Constraint_for_table: x3_input_acceleration >= -3.247708132667641
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6025358877983292 x3_input_acceleration + relaxation_CLF
   <= -20.14521979843397
 CBF_Constraint_obs2: - 0.1238400127582718 x1_input_acceleration
   - 0.3119678579286593 x2_input_acceleration
   + 1.20588616462013 x3_input_acceleration >= -1.908527168953515
 CBF_Constraint_obs3: 0.2561599872417282 x1_input_acceleration
   - 0.3119678579286593 x2_input_acceleration
   + 1.20588616462013 x3_input_acceleration >= -1.971379156833156
 CBF_Constraint_obs1: - 0.443866238801973 x1_input_acceleration
   - 0.3118845306352633 x2_input_acceleration
   + 1.205897012665479 x3_input_acceleration >= -2.135596004441643
 CBF_Constraint_for_table: x3_input_acceleration >= -3.254335536844056
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6038582095874392 x3_input_acceleration + relaxation_CLF
   <= -20.14585789491446
 CBF_Constraint_obs2: - 0.1237041141681839 x1_input_acceleration
   - 0.3114787422136325 x2_input_acceleration
   + 1.208877468812363 x3_input_acceleration >= -1.917133061644135
 CBF_Constraint_obs3: 0.2562958858318161 x1_input_acceleration
   - 0.3114787422136325 x2_input_acceleration
   + 1.208877468812363 x3_input_acceleration >= -1.98011415318436
 CBF_Constraint_obs1: - 0.4437041141681839 x1_input_acceleration
   - 0.3114787422136325 x2_input_acceleration
   + 1.208877468812363 x3_input_acceleration >= -2.144096352978682
 CBF_Constraint_for_table: x3_input_acceleration >= -3.260914661258203
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6051805313765491 x3_input_acceleration + relaxation_CLF
   <= -20.14649739022288
 CBF_Constraint_obs2: - 0.1237775470905472 x1_input_acceleration
   - 0.3115265451387423 x2_input_acceleration
   + 1.211468110582417 x3_input_acceleration >= -1.925030815559955
 CBF_Constraint_obs3: 0.2562830112285446 x1_input_acceleration
   - 0.3114885173753756 x2_input_acceleration
   + 1.211491952440327 x3_input_acceleration >= -1.988023536410944
 CBF_Constraint_obs1: - 0.4437169887714554 x1_input_acceleration
   - 0.3114885173753756 x2_input_acceleration
   + 1.211491952440327 x3_input_acceleration >= -2.152028266760991
 CBF_Constraint_for_table: x3_input_acceleration >= -3.267609120659991
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6065028531656591 x3_input_acceleration + relaxation_CLF
   <= -20.14713828435923
 CBF_Constraint_obs2: - 0.123708882539766 x1_input_acceleration
   - 0.3114941202119845 x2_input_acceleration
   + 1.214127178484672 x3_input_acceleration >= -1.933046600101284
 CBF_Constraint_obs3: 0.256291117460234 x1_input_acceleration
   - 0.3114941202119845 x2_input_acceleration
   + 1.214127178484672 x3_input_acceleration >= -1.996023161688506
 CBF_Constraint_obs1: - 0.4437775470905472 x1_input_acceleration
   - 0.3115265451387423 x2_input_acceleration
   + 1.214112754160637 x3_input_acceleration >= -2.160071349304138
 CBF_Constraint_for_table: x3_input_acceleration >= -3.274551833407267
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6078251749547691 x3_input_acceleration + relaxation_CLF
   <= -20.14778057732352
 CBF_Constraint_obs2: - 0.1237765934162308 x1_input_acceleration
   - 0.3115329824403781 x2_input_acceleration
   + 1.216734747973842 x3_input_acceleration >= -1.941021113941116
 CBF_Constraint_obs3: 0.2562234065837692 x1_input_acceleration
   - 0.3115329824403781 x2_input_acceleration
   + 1.216734747973842 x3_input_acceleration >= -2.003933350195696
 CBF_Constraint_obs1: - 0.4437765934162308 x1_input_acceleration
   - 0.3115329824403781 x2_input_acceleration
   + 1.216734747973842 x3_input_acceleration >= -2.1680423886741
 CBF_Constraint_for_table: x3_input_acceleration >= -3.280739951351688
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.609147496743879 x3_input_acceleration + relaxation_CLF
   <= -20.14842426911573
 CBF_Constraint_obs2: - 0.1238772060566117 x1_input_acceleration
   - 0.3118666492418307 x2_input_acceleration
   + 1.219112839580626 x3_input_acceleration >= -1.948553105900071
 CBF_Constraint_obs3: 0.2562234065837692 x1_input_acceleration
   - 0.3115329824403781 x2_input_acceleration
   + 1.219379391552062 x3_input_acceleration >= -2.011986667214331
 CBF_Constraint_obs1: - 0.4437765934162308 x1_input_acceleration
   - 0.3115329824403781 x2_input_acceleration
   + 1.219379391552062 x3_input_acceleration >= -2.176095705692735
 CBF_Constraint_for_table: x3_input_acceleration >= -3.287357222738492
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.610469818532989 x3_input_acceleration + relaxation_CLF
   <= -20.14906935973588
 CBF_Constraint_obs2: - 0.1238772060566117 x1_input_acceleration
   - 0.3118666492418307 x2_input_acceleration
   + 1.221757483158846 x3_input_acceleration >= -1.956622145930447
 CBF_Constraint_obs3: 0.2561227939433883 x1_input_acceleration
   - 0.3118666492418307 x2_input_acceleration
   + 1.221757483158846 x3_input_acceleration >= -2.019438800176666
 CBF_Constraint_obs1: - 0.4438772060566117 x1_input_acceleration
   - 0.3118666492418307 x2_input_acceleration
   + 1.221757483158846 x3_input_acceleration >= -2.183723910775736
 CBF_Constraint_for_table: x3_input_acceleration >= -3.293968831684042
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6117921403220989 x3_input_acceleration + relaxation_CLF
   <= -20.14971584918396
 CBF_Constraint_obs2: - 0.1237799312763382 x1_input_acceleration
   - 0.3118647418931979 x2_input_acceleration
   + 1.224407252736517 x3_input_acceleration >= -1.964692761470838
 CBF_Constraint_obs3: 0.2562195918865036 x1_input_acceleration
   - 0.3118906103090304 x2_input_acceleration
   + 1.224482950635382 x3_input_acceleration >= -2.027853410577695
 CBF_Constraint_obs1: - 0.4437804081134964 x1_input_acceleration
   - 0.3118906103090304 x2_input_acceleration
   + 1.224482950635382 x3_input_acceleration >= -2.191969124776314
 CBF_Constraint_for_table: x3_input_acceleration >= -3.300645409692397
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6131144621112089 x3_input_acceleration + relaxation_CLF
   <= -20.15036373745997
 CBF_Constraint_obs2: - 0.1237799312763382 x1_input_acceleration
   - 0.3118647418931979 x2_input_acceleration
   + 1.227051896314737 x3_input_acceleration >= -1.972796806090597
 CBF_Constraint_obs3: 0.2562200687236618 x1_input_acceleration
   - 0.3118647418931979 x2_input_acceleration
   + 1.227051896314737 x3_input_acceleration >= -2.035705871378076
 CBF_Constraint_obs1: - 0.4437799312763382 x1_input_acceleration
   - 0.3118647418931979 x2_input_acceleration
   + 1.227051896314737 x3_input_acceleration >= -2.199820751111667
 CBF_Constraint_for_table: x3_input_acceleration >= -3.307257018637946
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6144367839003189 x3_input_acceleration + relaxation_CLF
   <= -20.15101302456391
 CBF_Constraint_obs2: - 0.123713650911348 x1_input_acceleration
   - 0.3114723049119967 x2_input_acceleration
   + 1.230024246229932 x3_input_acceleration >= -1.981599638078148
 CBF_Constraint_obs3: 0.256286349088652 x1_input_acceleration
   - 0.3114723049119967 x2_input_acceleration
   + 1.230024246229932 x3_input_acceleration >= -2.044571669712368
 CBF_Constraint_obs1: - 0.4437799312763382 x1_input_acceleration
   - 0.3118647418931979 x2_input_acceleration
   + 1.229696539892957 x3_input_acceleration >= -2.207942281080566
 CBF_Constraint_for_table: x3_input_acceleration >= -3.314250097310059
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6157591056894288 x3_input_acceleration + relaxation_CLF
   <= -20.15166371049578
 CBF_Constraint_obs2: - 0.1238452579670121 x1_input_acceleration
   - 0.3118785701707858 x2_input_acceleration
   + 1.232343210029099 x3_input_acceleration >= -1.989094597196892
 CBF_Constraint_obs3: 0.256286349088652 x1_input_acceleration
   - 0.3114723049119967 x2_input_acceleration
   + 1.232668889808152 x3_input_acceleration >= -2.052712851696554
 CBF_Constraint_obs1: - 0.443713650911348 x1_input_acceleration
   - 0.3114723049119967 x2_input_acceleration
   + 1.232668889808152 x3_input_acceleration >= -2.216711740791413
 CBF_Constraint_for_table: x3_input_acceleration >= -3.320384273050958
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6170814274785388 x3_input_acceleration + relaxation_CLF
   <= -20.15231579525558
 CBF_Constraint_obs2: - 0.1238452579670121 x1_input_acceleration
   - 0.3118785701707858 x2_input_acceleration
   + 1.234987853607319 x3_input_acceleration >= -1.997251111262877
 CBF_Constraint_obs3: 0.2561547420329879 x1_input_acceleration
   - 0.3118785701707858 x2_input_acceleration
   + 1.234987853607319 x3_input_acceleration >= -2.060098116194216
 CBF_Constraint_obs1: - 0.443713650911348 x1_input_acceleration
   - 0.3114723049119967 x2_input_acceleration
   + 1.235313533386372 x3_input_acceleration >= -2.22487040812474
 CBF_Constraint_for_table: x3_input_acceleration >= -3.327039989433641
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6184037492676487 x3_input_acceleration + relaxation_CLF
   <= -20.15296927884332
 CBF_Constraint_obs2: - 0.1237785007648636 x1_input_acceleration
   - 0.3115427576021212 x2_input_acceleration
   + 1.237897380226921 x3_input_acceleration >= -2.005962413798255
 CBF_Constraint_obs3: 0.2562214992351364 x1_input_acceleration
   - 0.3115427576021212 x2_input_acceleration
   + 1.237897380226921 x3_input_acceleration >= -2.068872838071635
 CBF_Constraint_obs1: - 0.4437785007648636 x1_input_acceleration
   - 0.3115427576021212 x2_input_acceleration
   + 1.237897380226921 x3_input_acceleration >= -2.232985214410147
 CBF_Constraint_for_table: x3_input_acceleration >= -3.333650406286295
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6197260710567587 x3_input_acceleration + relaxation_CLF
   <= -20.15362416125898
 CBF_Constraint_obs2: - 0.1237789776020218 x1_input_acceleration
   - 0.3115154586748141 x2_input_acceleration
   + 1.240562885430812 x3_input_acceleration >= -2.014199236247504
 CBF_Constraint_obs3: 0.2562210223979782 x1_input_acceleration
   - 0.3115154586748141 x2_input_acceleration
   + 1.240562885430812 x3_input_acceleration >= -2.077109207525583
 CBF_Constraint_obs1: - 0.4437789776020218 x1_input_acceleration
   - 0.3115154586748141 x2_input_acceleration
   + 1.240562885430812 x3_input_acceleration >= -2.241222418329122
 CBF_Constraint_for_table: x3_input_acceleration >= -3.340262015231845
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6210483928458687 x3_input_acceleration + relaxation_CLF
   <= -20.15428044250257
 CBF_Constraint_obs2: - 0.1238257076435257 x1_input_acceleration
   - 0.3119443736986178 x2_input_acceleration
   + 1.242909028947997 x3_input_acceleration >= -2.021831190495119
 CBF_Constraint_obs3: 0.2561742923564743 x1_input_acceleration
   - 0.3119443736986178 x2_input_acceleration
   + 1.242909028947997 x3_input_acceleration >= -2.084696768233769
 CBF_Constraint_obs1: - 0.4437789776020218 x1_input_acceleration
   - 0.3115154586748141 x2_input_acceleration
   + 1.243207529009032 x3_input_acceleration >= -2.249433277674523
 CBF_Constraint_for_table: x3_input_acceleration >= -3.346885843129574
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6223707146349786 x3_input_acceleration + relaxation_CLF
   <= -20.1549381225741
 CBF_Constraint_obs2: - 0.1238547947101761 x1_input_acceleration
   - 0.3118672452882785 x2_input_acceleration
   + 1.245576083872652 x3_input_acceleration >= -2.030076211964914
 CBF_Constraint_obs3: 0.2561437747783493 x1_input_acceleration
   - 0.3118932329134005 x2_input_acceleration
   + 1.245547712061739 x3_input_acceleration >= -2.092845156396384
 CBF_Constraint_obs1: - 0.4438562252216507 x1_input_acceleration
   - 0.3118932329134005 x2_input_acceleration
   + 1.245547712061739 x3_input_acceleration >= -2.257093550534273
 CBF_Constraint_for_table: x3_input_acceleration >= -3.353468841845632
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6236930364240886 x3_input_acceleration + relaxation_CLF
   <= -20.15559720147356
 CBF_Constraint_obs2: - 0.1237732555561234 x1_input_acceleration
   - 0.3115346513704318 x2_input_acceleration
   + 1.248482272632147 x3_input_acceleration >= -2.038852053590234
 CBF_Constraint_obs3: 0.2562267444438766 x1_input_acceleration
   - 0.3115346513704318 x2_input_acceleration
   + 1.248482272632147 x3_input_acceleration >= -2.101767460811917
 CBF_Constraint_obs1: - 0.4437732555561234 x1_input_acceleration
   - 0.3115346513704318 x2_input_acceleration
   + 1.248482272632147 x3_input_acceleration >= -2.265870658035133
 CBF_Constraint_for_table: x3_input_acceleration >= -3.360144227761091
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6250153582131985 x3_input_acceleration + relaxation_CLF
   <= -20.15625767920095
 CBF_Constraint_obs2: - 0.1237823154621293 x1_input_acceleration
   - 0.3118561588243502 x2_input_acceleration
   + 1.250861913959695 x3_input_acceleration >= -2.046539816515945
 CBF_Constraint_obs3: 0.2562176845378707 x1_input_acceleration
   - 0.3118561588243502 x2_input_acceleration
   + 1.250861913959695 x3_input_acceleration >= -2.109446616826922
 CBF_Constraint_obs1: - 0.4437823154621293 x1_input_acceleration
   - 0.3118561588243502 x2_input_acceleration
   + 1.250861913959695 x3_input_acceleration >= -2.273565668885648
 CBF_Constraint_for_table: x3_input_acceleration >= -3.367075019579413
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6263376800023085 x3_input_acceleration + relaxation_CLF
   <= -20.15691955575627
 CBF_Constraint_obs2: - 0.1236979152851273 x1_input_acceleration
   - 0.3114867292360324 x2_input_acceleration
   + 1.253797905041577 x3_input_acceleration >= -2.055417929278369
 CBF_Constraint_obs3: 0.2562176845378707 x1_input_acceleration
   - 0.3118561588243502 x2_input_acceleration
   + 1.253506557537915 x3_input_acceleration >= -2.117725569321475
 CBF_Constraint_obs1: - 0.4437823154621293 x1_input_acceleration
   - 0.3118561588243502 x2_input_acceleration
   + 1.253506557537915 x3_input_acceleration >= -2.281844621380202
 CBF_Constraint_for_table: x3_input_acceleration >= -3.373328702633087
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6276600017914185 x3_input_acceleration + relaxation_CLF
   <= -20.15758283113952
 CBF_Constraint_obs2: - 0.1237618114643265 x1_input_acceleration
   - 0.3119053922609347 x2_input_acceleration
   + 1.256181122647812 x3_input_acceleration >= -2.06324121574387
 CBF_Constraint_obs3: 0.2563020847148727 x1_input_acceleration
   - 0.3114867292360324 x2_input_acceleration
   + 1.256442548619797 x3_input_acceleration >= -2.126703273876953
 CBF_Constraint_obs1: - 0.4436979152851273 x1_input_acceleration
   - 0.3114867292360324 x2_input_acceleration
   + 1.256442548619797 x3_input_acceleration >= -2.290674625625925
 CBF_Constraint_for_table: x3_input_acceleration >= -3.379912893442039
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6289823235805284 x3_input_acceleration + relaxation_CLF
   <= -20.1582475053507
 CBF_Constraint_obs2: - 0.1237618114643265 x1_input_acceleration
   - 0.3119053922609347 x2_input_acceleration
   + 1.258825766226032 x3_input_acceleration >= -2.07155533676617
 CBF_Constraint_obs3: 0.2562381885356735 x1_input_acceleration
   - 0.3119053922609347 x2_input_acceleration
   + 1.258825766226032 x3_input_acceleration >= -2.134481615875059
 CBF_Constraint_obs1: - 0.4437618114643265 x1_input_acceleration
   - 0.3119053922609347 x2_input_acceleration
   + 1.258825766226032 x3_input_acceleration >= -2.298564785937631
 CBF_Constraint_for_table: x3_input_acceleration >= -3.386561755290574
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6303046453696384 x3_input_acceleration + relaxation_CLF
   <= -20.15891357838982
 CBF_Constraint_obs2: - 0.1237661029987503 x1_input_acceleration
   - 0.3115336976961154 x2_input_acceleration
   + 1.261713119917777 x3_input_acceleration >= -2.080364112530455
 CBF_Constraint_obs3: 0.2562338970012497 x1_input_acceleration
   - 0.3115336976961154 x2_input_acceleration
   + 1.261713119917777 x3_input_acceleration >= -2.143286314681641
 CBF_Constraint_obs1: - 0.4437661029987504 x1_input_acceleration
   - 0.3115336976961154 x2_input_acceleration
   + 1.261713119917777 x3_input_acceleration >= -2.307376994929455
 CBF_Constraint_for_table: x3_input_acceleration >= -3.393192735745676
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6316269671587483 x3_input_acceleration + relaxation_CLF
   <= -20.15958105025686
 CBF_Constraint_obs2: - 0.1234232570820023 x1_input_acceleration
   - 0.3117277704195041 x2_input_acceleration
   + 1.264243203368739 x3_input_acceleration >= -2.08839797562937
 CBF_Constraint_obs3: 0.2565767429179977 x1_input_acceleration
   - 0.3117277704195041 x2_input_acceleration
   + 1.264243203368739 x3_input_acceleration >= -2.151645881401468
 CBF_Constraint_obs1: - 0.4434232570820023 x1_input_acceleration
   - 0.3117277704195041 x2_input_acceleration
   + 1.264243203368739 x3_input_acceleration >= -2.315136581294972
 CBF_Constraint_for_table: x3_input_acceleration >= -3.399804344691226
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6329492889478583 x3_input_acceleration + relaxation_CLF
   <= -20.16024992095184
 CBF_Constraint_obs2: - 0.1237031604938675 x1_input_acceleration
   - 0.3114869676546115 x2_input_acceleration
   + 1.267031970978026 x3_input_acceleration >= -2.097120773019121
 CBF_Constraint_obs3: 0.2562968395061325 x1_input_acceleration
   - 0.3114869676546115 x2_input_acceleration
   + 1.267031970978026 x3_input_acceleration >= -2.160102770549947
 CBF_Constraint_obs1: - 0.4434232570820023 x1_input_acceleration
   - 0.3117277704195041 x2_input_acceleration
   + 1.266887846946959 x3_input_acceleration >= -2.323504005642285
 CBF_Constraint_for_table: x3_input_acceleration >= -3.406390025616298
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6342716107369682 x3_input_acceleration + relaxation_CLF
   <= -20.16092019047475
 CBF_Constraint_obs2: - 0.12377945443918 x1_input_acceleration
   - 0.3115168891862887 x2_input_acceleration
   + 1.269652415070467 x3_input_acceleration >= -2.105476725854791
 CBF_Constraint_obs3: 0.25622054556082 x1_input_acceleration
   - 0.3115168891862887 x2_input_acceleration
   + 1.269652415070467 x3_input_acceleration >= -2.16838624413757
 CBF_Constraint_obs1: - 0.44377945443918 x1_input_acceleration
   - 0.3115168891862887 x2_input_acceleration
   + 1.269652415070467 x3_input_acceleration >= -2.332500289406135
 CBF_Constraint_for_table: x3_input_acceleration >= -3.413425423586201
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6355939325260782 x3_input_acceleration + relaxation_CLF
   <= -20.16159185882558
 CBF_Constraint_obs2: - 0.12377945443918 x1_input_acceleration
   - 0.3115168891862887 x2_input_acceleration
   + 1.272297058648687 x3_input_acceleration >= -2.113879913794579
 CBF_Constraint_obs3: 0.25622054556082 x1_input_acceleration
   - 0.3115168891862887 x2_input_acceleration
   + 1.272297058648687 x3_input_acceleration >= -2.176789432077358
 CBF_Constraint_obs1: - 0.44377945443918 x1_input_acceleration
   - 0.3115168891862887 x2_input_acceleration
   + 1.272297058648687 x3_input_acceleration >= -2.340903477345923
 CBF_Constraint_for_table: x3_input_acceleration >= -3.420037032531751
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6369162543151882 x3_input_acceleration + relaxation_CLF
   <= -20.16226492600435
 CBF_Constraint_obs2: - 0.1237813617878128 x1_input_acceleration
   - 0.3115296445802707 x2_input_acceleration
   + 1.274941463808328 x3_input_acceleration >= -2.122310351395095
 CBF_Constraint_obs3: 0.2562186382121872 x1_input_acceleration
   - 0.3115296445802707 x2_input_acceleration
   + 1.274941463808328 x3_input_acceleration >= -2.185218057696673
 CBF_Constraint_obs1: - 0.4437813617878129 x1_input_acceleration
   - 0.3115296445802707 x2_input_acceleration
   + 1.274941463808328 x3_input_acceleration >= -2.349335440825346
 CBF_Constraint_for_table: x3_input_acceleration >= -3.426629866014197
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6382385761042981 x3_input_acceleration + relaxation_CLF
   <= -20.16293939201106
 CBF_Constraint_obs2: - 0.1237761165790726 x1_input_acceleration
   - 0.3115331016496676 x2_input_acceleration
   + 1.277580742968518 x3_input_acceleration >= -2.13073244407856
 CBF_Constraint_obs3: 0.2562238834209274 x1_input_acceleration
   - 0.3115331016496676 x2_input_acceleration
   + 1.277580742968518 x3_input_acceleration >= -2.193645133328441
 CBF_Constraint_obs1: - 0.4437761165790726 x1_input_acceleration
   - 0.3115331016496676 x2_input_acceleration
   + 1.277580742968518 x3_input_acceleration >= -2.357753337341818
 CBF_Constraint_for_table: x3_input_acceleration >= -3.433173823687926
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6395608978934081 x3_input_acceleration + relaxation_CLF
   <= -20.16361525684569
 CBF_Constraint_obs2: - 0.1236936237507035 x1_input_acceleration
   - 0.311498769374277 x2_input_acceleration
   + 1.280254235194809 x3_input_acceleration >= -2.139228128265644
 CBF_Constraint_obs3: 0.2563063762492965 x1_input_acceleration
   - 0.311498769374277 x2_input_acceleration
   + 1.280254235194809 x3_input_acceleration >= -2.202219185702475
 CBF_Constraint_obs1: - 0.4436936237507035 x1_input_acceleration
   - 0.311498769374277 x2_input_acceleration
   + 1.280254235194809 x3_input_acceleration >= -2.366183027266206
 CBF_Constraint_for_table: x3_input_acceleration >= -3.439404558953361
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.640883219682518 x3_input_acceleration + relaxation_CLF
   <= -20.16429252050825
 CBF_Constraint_obs2: - 0.123833337038057 x1_input_acceleration
   - 0.3118792854265231 x2_input_acceleration
   + 1.282586908062274 x3_input_acceleration >= -2.147040700966106
 CBF_Constraint_obs3: 0.256166662961943 x1_input_acceleration
   - 0.3118792854265231 x2_input_acceleration
   + 1.282586908062274 x3_input_acceleration >= -2.209899030779952
 CBF_Constraint_obs1: - 0.443833337038057 x1_input_acceleration
   - 0.3118792854265231 x2_input_acceleration
   + 1.282586908062274 x3_input_acceleration >= -2.374107370596552
 CBF_Constraint_for_table: x3_input_acceleration >= -3.446095144053238
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.642205541471628 x3_input_acceleration + relaxation_CLF
   <= -20.16497118299875
 CBF_Constraint_obs2: - 0.123833337038057 x1_input_acceleration
   - 0.3118792854265231 x2_input_acceleration
   + 1.285231551640494 x3_input_acceleration >= -2.155529406715466
 CBF_Constraint_obs3: 0.256166662961943 x1_input_acceleration
   - 0.3118792854265231 x2_input_acceleration
   + 1.285231551640494 x3_input_acceleration >= -2.218387736529311
 CBF_Constraint_obs1: - 0.443833337038057 x1_input_acceleration
   - 0.3118792854265231 x2_input_acceleration
   + 1.285231551640494 x3_input_acceleration >= -2.382596076345911
 CBF_Constraint_for_table: x3_input_acceleration >= -3.452655493004281
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.643527863260738 x3_input_acceleration + relaxation_CLF
   <= -20.16565124431717
 CBF_Constraint_obs2: - 0.1237775470905472 x1_input_acceleration
   - 0.3118497215227145 x2_input_acceleration
   + 1.28790909698263 x3_input_acceleration >= -2.164101215085202
 CBF_Constraint_obs3: 0.256166662961943 x1_input_acceleration
   - 0.3118792854265231 x2_input_acceleration
   + 1.287876195218714 x3_input_acceleration >= -2.22689392762781
 CBF_Constraint_obs1: - 0.443833337038057 x1_input_acceleration
   - 0.3118792854265231 x2_input_acceleration
   + 1.287876195218714 x3_input_acceleration >= -2.39110226744441
 CBF_Constraint_for_table: x3_input_acceleration >= -3.459267101949831
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6448501850498479 x3_input_acceleration + relaxation_CLF
   <= -20.16633270446353
 CBF_Constraint_obs2: - 0.1238023426227738 x1_input_acceleration
   - 0.3115236841157931 x2_input_acceleration
   + 1.290784172117552 x3_input_acceleration >= -2.173122255991446
 CBF_Constraint_obs3: 0.2562224529094528 x1_input_acceleration
   - 0.3118497215227145 x2_input_acceleration
   + 1.29055374056085 x3_input_acceleration >= -2.235536439330417
 CBF_Constraint_obs1: - 0.4437775470905472 x1_input_acceleration
   - 0.3118497215227145 x2_input_acceleration
   + 1.29055374056085 x3_input_acceleration >= -2.399647146738875
 CBF_Constraint_for_table: x3_input_acceleration >= -3.466316507011256
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6461725068389579 x3_input_acceleration + relaxation_CLF
   <= -20.16701556343782
 CBF_Constraint_obs2: - 0.1238023426227738 x1_input_acceleration
   - 0.3115236841157931 x2_input_acceleration
   + 1.293428815695772 x3_input_acceleration >= -2.181665158845163
 CBF_Constraint_obs3: 0.2561976573772262 x1_input_acceleration
   - 0.3115236841157931 x2_input_acceleration
   + 1.293428815695772 x3_input_acceleration >= -2.244552933353527
 CBF_Constraint_obs1: - 0.4438023426227738 x1_input_acceleration
   - 0.3115236841157931 x2_input_acceleration
   + 1.293428815695772 x3_input_acceleration >= -2.408707032943381
 CBF_Constraint_for_table: x3_input_acceleration >= -3.472862848870777
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6474948286280678 x3_input_acceleration + relaxation_CLF
   <= -20.16769982124003
 CBF_Constraint_obs2: - 0.1238023426227738 x1_input_acceleration
   - 0.3115236841157931 x2_input_acceleration
   + 1.296073459273992 x3_input_acceleration >= -2.190225547048018
 CBF_Constraint_obs3: 0.2561976573772262 x1_input_acceleration
   - 0.3115236841157931 x2_input_acceleration
   + 1.296073459273992 x3_input_acceleration >= -2.253113321556383
 CBF_Constraint_obs1: - 0.4438023426227738 x1_input_acceleration
   - 0.3115236841157931 x2_input_acceleration
   + 1.296073459273992 x3_input_acceleration >= -2.417267421146238
 CBF_Constraint_for_table: x3_input_acceleration >= -3.479175540522778
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6488171504171778 x3_input_acceleration + relaxation_CLF
   <= -20.16838547787019
 CBF_Constraint_obs2: - 0.1238595630817582 x1_input_acceleration
   - 0.3118803583101291 x2_input_acceleration
   + 1.298449166694985 x3_input_acceleration >= -2.1982259846967
 CBF_Constraint_obs3: 0.2563264034099411 x1_input_acceleration
   - 0.3115651689485568 x2_input_acceleration
   + 1.298676975647317 x3_input_acceleration >= -2.261672458329349
 CBF_Constraint_obs1: - 0.4436735965900589 x1_input_acceleration
   - 0.3115651689485568 x2_input_acceleration
   + 1.298676975647317 x3_input_acceleration >= -2.425601252361953
 CBF_Constraint_for_table: x3_input_acceleration >= -3.485704597035314
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6501394722062878 x3_input_acceleration + relaxation_CLF
   <= -20.16907253332827
 CBF_Constraint_obs2: - 0.1238595630817582 x1_input_acceleration
   - 0.3118803583101291 x2_input_acceleration
   + 1.301093810273205 x3_input_acceleration >= -2.206819565497132
 CBF_Constraint_obs3: 0.2561404369182418 x1_input_acceleration
   - 0.3118803583101291 x2_input_acceleration
   + 1.301093810273205 x3_input_acceleration >= -2.269652980569461
 CBF_Constraint_obs1: - 0.4438595630817582 x1_input_acceleration
   - 0.3118803583101291 x2_input_acceleration
   + 1.301093810273205 x3_input_acceleration >= -2.433907215962538
 CBF_Constraint_for_table: x3_input_acceleration >= -3.492316205980864
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6514617939953977 x3_input_acceleration + relaxation_CLF
   <= -20.16976098761428
 CBF_Constraint_obs2: - 0.1238414432697464 x1_input_acceleration
   - 0.3119199357942599 x2_input_acceleration
   + 1.303793409333908 x3_input_acceleration >= -2.215635004549455
 CBF_Constraint_obs3: 0.2561585567302536 x1_input_acceleration
   - 0.3119199357942599 x2_input_acceleration
   + 1.303793409333908 x3_input_acceleration >= -2.278485633443196
 CBF_Constraint_obs1: - 0.4438595630817582 x1_input_acceleration
   - 0.3118803583101291 x2_input_acceleration
   + 1.303738453851425 x3_input_acceleration >= -2.44251828211211
 CBF_Constraint_for_table: x3_input_acceleration >= -3.499027950729636
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6527841157845077 x3_input_acceleration + relaxation_CLF
   <= -20.17045084072823
 CBF_Constraint_obs2: - 0.1238714840107132 x1_input_acceleration
   - 0.3118976436571139 x2_input_acceleration
   + 1.306365335245502 x3_input_acceleration >= -2.224018342251211
 CBF_Constraint_obs3: 0.2561285159892868 x1_input_acceleration
   - 0.3118976436571139 x2_input_acceleration
   + 1.306365335245502 x3_input_acceleration >= -2.286840432441033
 CBF_Constraint_obs1: - 0.4438714840107132 x1_input_acceleration
   - 0.3118976436571139 x2_input_acceleration
   + 1.306365335245502 x3_input_acceleration >= -2.451115529459782
 CBF_Constraint_for_table: x3_input_acceleration >= -3.505639559675186
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6541064375736176 x3_input_acceleration + relaxation_CLF
   <= -20.1711420926701
 CBF_Constraint_obs2: - 0.1237165119342972 x1_input_acceleration
   - 0.3115006767229098 x2_input_acceleration
   + 1.309335062556327 x3_input_acceleration >= -2.23337094120425
 CBF_Constraint_obs3: 0.2562834880657028 x1_input_acceleration
   - 0.3115006767229098 x2_input_acceleration
   + 1.309335062556327 x3_input_acceleration >= -2.296340254866667
 CBF_Constraint_obs1: - 0.4437165119342972 x1_input_acceleration
   - 0.3115006767229098 x2_input_acceleration
   + 1.309335062556327 x3_input_acceleration >= -2.460344150751688
 CBF_Constraint_for_table: x3_input_acceleration >= -3.512193948161752
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6554287593627276 x3_input_acceleration + relaxation_CLF
   <= -20.17183474343991
 CBF_Constraint_obs2: - 0.1237165119342972 x1_input_acceleration
   - 0.3115006767229098 x2_input_acceleration
   + 1.311979706134547 x3_input_acceleration >= -2.242036495291139
 CBF_Constraint_obs3: 0.2562834880657028 x1_input_acceleration
   - 0.3115006767229098 x2_input_acceleration
   + 1.311979706134547 x3_input_acceleration >= -2.305005808953557
 CBF_Constraint_obs1: - 0.4437165119342972 x1_input_acceleration
   - 0.3115006767229098 x2_input_acceleration
   + 1.311979706134547 x3_input_acceleration >= -2.469009704838577
 CBF_Constraint_for_table: x3_input_acceleration >= -3.518805557107301
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6567510811518376 x3_input_acceleration + relaxation_CLF
   <= -20.17252879303764
 CBF_Constraint_obs2: - 0.1236735965900589 x1_input_acceleration
   - 0.3115441881135959 x2_input_acceleration
   + 1.314671556591429 x3_input_acceleration >= -2.250895301689263
 CBF_Constraint_obs3: 0.2562834880657028 x1_input_acceleration
   - 0.3115006767229098 x2_input_acceleration
   + 1.314624349712767 x3_input_acceleration >= -2.313688848389585
 CBF_Constraint_obs1: - 0.4437165119342972 x1_input_acceleration
   - 0.3115006767229098 x2_input_acceleration
   + 1.314624349712767 x3_input_acceleration >= -2.477692744274606
 CBF_Constraint_for_table: x3_input_acceleration >= -3.525364713965449
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6580734029409475 x3_input_acceleration + relaxation_CLF
   <= -20.17322424146331
 CBF_Constraint_obs2: - 0.1236735965900589 x1_input_acceleration
   - 0.3115441881135959 x2_input_acceleration
   + 1.317316200169649 x3_input_acceleration >= -2.259596138587852
 CBF_Constraint_obs3: 0.2563264034099411 x1_input_acceleration
   - 0.3115441881135959 x2_input_acceleration
   + 1.317316200169649 x3_input_acceleration >= -2.322606221827296
 CBF_Constraint_obs1: - 0.4436735965900589 x1_input_acceleration
   - 0.3115441881135959 x2_input_acceleration
   + 1.317316200169649 x3_input_acceleration >= -2.4865350158599
 CBF_Constraint_for_table: x3_input_acceleration >= -3.532423357746814
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6593957247300575 x3_input_acceleration + relaxation_CLF
   <= -20.17392108871691
 CBF_Constraint_obs2: - 0.1237780239277054 x1_input_acceleration
   - 0.31185091361561 x2_input_acceleration
   + 1.319651614850774 x3_input_acceleration >= -2.267565470135741
 CBF_Constraint_obs3: 0.2562219760722946 x1_input_acceleration
   - 0.31185091361561 x2_input_acceleration
   + 1.319651614850774 x3_input_acceleration >= -2.330476347404421
 CBF_Constraint_obs1: - 0.4437751629047562 x1_input_acceleration
   - 0.3118451915697116 x2_input_acceleration
   + 1.319656502431645 x3_input_acceleration >= -2.494596378908021
 CBF_Constraint_for_table: x3_input_acceleration >= -3.539076093897259
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6607180465191674 x3_input_acceleration + relaxation_CLF
   <= -20.17461933479844
 CBF_Constraint_obs2: - 0.1237961437397171 x1_input_acceleration
   - 0.3118455491975802 x2_input_acceleration
   + 1.322291609266701 x3_input_acceleration >= -2.276285289620359
 CBF_Constraint_obs3: 0.2562038562602829 x1_input_acceleration
   - 0.3118455491975802 x2_input_acceleration
   + 1.322291609266701 x3_input_acceleration >= -2.339178953067628
 CBF_Constraint_obs1: - 0.4437961437397172 x1_input_acceleration
   - 0.3118455491975802 x2_input_acceleration
   + 1.322291609266701 x3_input_acceleration >= -2.503322204612133
 CBF_Constraint_for_table: x3_input_acceleration >= -3.545611408897496
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6620403683082774 x3_input_acceleration + relaxation_CLF
   <= -20.1753189797079
 CBF_Constraint_obs2: - 0.123871007173555 x1_input_acceleration
   - 0.3119007430986422 x2_input_acceleration
   + 1.324866157782665 x3_input_acceleration >= -2.284870545001848
 CBF_Constraint_obs3: 0.256128992826445 x1_input_acceleration
   - 0.3119007430986422 x2_input_acceleration
   + 1.324866157782665 x3_input_acceleration >= -2.347693088186971
 CBF_Constraint_obs1: - 0.443871007173555 x1_input_acceleration
   - 0.3119007430986422 x2_input_acceleration
   + 1.324866157782665 x3_input_acceleration >= -2.511967350740692
 CBF_Constraint_for_table: x3_input_acceleration >= -3.551866582067289
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6633626900973874 x3_input_acceleration + relaxation_CLF
   <= -20.1760200234453
 CBF_Constraint_obs2: - 0.123871007173555 x1_input_acceleration
   - 0.3119007430986422 x2_input_acceleration
   + 1.327510801360885 x3_input_acceleration >= -2.29363878461687
 CBF_Constraint_obs3: 0.256128992826445 x1_input_acceleration
   - 0.3119007430986422 x2_input_acceleration
   + 1.327510801360885 x3_input_acceleration >= -2.356461327801993
 CBF_Constraint_obs1: - 0.443871007173555 x1_input_acceleration
   - 0.3119007430986422 x2_input_acceleration
   + 1.327510801360885 x3_input_acceleration >= -2.520735590355714
 CBF_Constraint_for_table: x3_input_acceleration >= -3.558478191012839
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6646850118864973 x3_input_acceleration + relaxation_CLF
   <= -20.17672246601062
 CBF_Constraint_obs2: - 0.1238543178730179 x1_input_acceleration
   - 0.3118902526811618 x2_input_acceleration
   + 1.330203367073505 x3_input_acceleration >= -2.302570524433747
 CBF_Constraint_obs3: 0.2561456821269821 x1_input_acceleration
   - 0.3118902526811618 x2_input_acceleration
   + 1.330203367073505 x3_input_acceleration >= -2.36540892245438
 CBF_Constraint_obs1: - 0.4438543178730179 x1_input_acceleration
   - 0.3118902526811618 x2_input_acceleration
   + 1.330203367073505 x3_input_acceleration >= -2.529653978732162
 CBF_Constraint_for_table: x3_input_acceleration >= -3.565427758294265
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6660073336756073 x3_input_acceleration + relaxation_CLF
   <= -20.17742630740388
 CBF_Constraint_obs2: - 0.1237813617878128 x1_input_acceleration
   - 0.3115305982545871 x2_input_acceleration
   + 1.333136735551016 x3_input_acceleration >= -2.312033368563159
 CBF_Constraint_obs3: 0.2563006542033981 x1_input_acceleration
   - 0.3114943586305636 x2_input_acceleration
   + 1.333158789269583 x3_input_acceleration >= -2.375038897763122
 CBF_Constraint_obs1: - 0.4436993457966019 x1_input_acceleration
   - 0.3114943586305636 x2_input_acceleration
   + 1.333158789269583 x3_input_acceleration >= -2.539012752907175
 CBF_Constraint_for_table: x3_input_acceleration >= -3.572039367239815
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6673296554647172 x3_input_acceleration + relaxation_CLF
   <= -20.17813154762506
 CBF_Constraint_obs2: - 0.1237813617878128 x1_input_acceleration
   - 0.3115305982545871 x2_input_acceleration
   + 1.335781379129236 x3_input_acceleration >= -2.320856290004139
 CBF_Constraint_obs3: 0.2562186382121872 x1_input_acceleration
   - 0.3115305982545871 x2_input_acceleration
   + 1.335781379129236 x3_input_acceleration >= -2.383763996305716
 CBF_Constraint_obs1: - 0.4437813617878129 x1_input_acceleration
   - 0.3115305982545871 x2_input_acceleration
   + 1.335781379129236 x3_input_acceleration >= -2.547881379434389
 CBF_Constraint_for_table: x3_input_acceleration >= -3.578718925480409
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6686519772538272 x3_input_acceleration + relaxation_CLF
   <= -20.17883818667418
 CBF_Constraint_obs2: - 0.1237842228107621 x1_input_acceleration
   - 0.3115384660676974 x2_input_acceleration
   + 1.338406591593259 x3_input_acceleration >= -2.329638692596205
 CBF_Constraint_obs3: 0.2562157771892379 x1_input_acceleration
   - 0.3115384660676974 x2_input_acceleration
   + 1.338406591593259 x3_input_acceleration >= -2.392543680925981
 CBF_Constraint_obs1: - 0.4437842228107621 x1_input_acceleration
   - 0.3115384660676974 x2_input_acceleration
   + 1.338406591593259 x3_input_acceleration >= -2.556666070844815
 CBF_Constraint_for_table: x3_input_acceleration >= -3.584855485407099
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6699742990429371 x3_input_acceleration + relaxation_CLF
   <= -20.17954622455123
 CBF_Constraint_obs2: - 0.1233460094623734 x1_input_acceleration
   - 0.3114204488710421 x2_input_acceleration
   + 1.341218008967561 x3_input_acceleration >= -2.338828352005302
 CBF_Constraint_obs3: 0.2566539905376266 x1_input_acceleration
   - 0.3114204488710421 x2_input_acceleration
   + 1.341218008967561 x3_input_acceleration >= -2.402149643016047
 CBF_Constraint_obs1: - 0.4437842228107621 x1_input_acceleration
   - 0.3115384660676974 x2_input_acceleration
   + 1.341051235171479 x3_input_acceleration >= -2.565523834513145
 CBF_Constraint_for_table: x3_input_acceleration >= -3.59188998930733
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6712966208320471 x3_input_acceleration + relaxation_CLF
   <= -20.18025566125621
 CBF_Constraint_obs2: - 0.1238671924762894 x1_input_acceleration
   - 0.3118963323549289 x2_input_acceleration
   + 1.343380093341679 x3_input_acceleration >= -2.346615598368892
 CBF_Constraint_obs3: 0.2561328075237106 x1_input_acceleration
   - 0.3118963323549289 x2_input_acceleration
   + 1.343380093341679 x3_input_acceleration >= -2.409441765516417
 CBF_Constraint_obs1: - 0.4438671924762894 x1_input_acceleration
   - 0.3118963323549289 x2_input_acceleration
   + 1.343380093341679 x3_input_acceleration >= -2.573709352349923
 CBF_Constraint_for_table: x3_input_acceleration >= -3.598165130033122
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6726189426211571 x3_input_acceleration + relaxation_CLF
   <= -20.18096649678912
 CBF_Constraint_obs2: - 0.1238676693134476 x1_input_acceleration
   - 0.3118691526369113 x2_input_acceleration
   + 1.346062407055397 x3_input_acceleration >= -2.355611964444226
 CBF_Constraint_obs3: 0.2561323306865524 x1_input_acceleration
   - 0.3118691526369113 x2_input_acceleration
   + 1.346062407055397 x3_input_acceleration >= -2.418437678596451
 CBF_Constraint_obs1: - 0.4438676693134476 x1_input_acceleration
   - 0.3118691526369113 x2_input_acceleration
   + 1.346062407055397 x3_input_acceleration >= -2.582706099894984
 CBF_Constraint_for_table: x3_input_acceleration >= -3.604726969100285
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.673941264410267 x3_input_acceleration + relaxation_CLF
   <= -20.18167873114997
 CBF_Constraint_obs2: - 0.123800435274141 x1_input_acceleration
   - 0.3118467412904757 x2_input_acceleration
   + 1.348718017888256 x3_input_acceleration >= -2.364519037004977
 CBF_Constraint_obs3: 0.256199564725859 x1_input_acceleration
   - 0.3118467412904757 x2_input_acceleration
   + 1.348718017888256 x3_input_acceleration >= -2.427408623494543
 CBF_Constraint_obs1: - 0.443800435274141 x1_input_acceleration
   - 0.3118467412904757 x2_input_acceleration
   + 1.348718017888256 x3_input_acceleration >= -2.59155938522429
 CBF_Constraint_for_table: x3_input_acceleration >= -3.611303709328641
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.675263586199377 x3_input_acceleration + relaxation_CLF
   <= -20.18239236433874
 CBF_Constraint_obs2: - 0.123800435274141 x1_input_acceleration
   - 0.3118467412904757 x2_input_acceleration
   + 1.351362661466476 x3_input_acceleration >= -2.37344497579164
 CBF_Constraint_obs3: 0.256199564725859 x1_input_acceleration
   - 0.3118467412904757 x2_input_acceleration
   + 1.351362661466476 x3_input_acceleration >= -2.436334562281206
 CBF_Constraint_obs1: - 0.443800435274141 x1_input_acceleration
   - 0.3118467412904757 x2_input_acceleration
   + 1.351362661466476 x3_input_acceleration >= -2.600485324010953
 CBF_Constraint_for_table: x3_input_acceleration >= -3.618393049502066
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6765859079884869 x3_input_acceleration + relaxation_CLF
   <= -20.18310739635544
 CBF_Constraint_obs2: - 0.1237870838337113 x1_input_acceleration
   - 0.3115542016939181 x2_input_acceleration
   + 1.354253829855487 x3_input_acceleration >= -2.382990872994303
 CBF_Constraint_obs3: 0.25615951040457 x1_input_acceleration
   - 0.3119433008150119 x2_input_acceleration
   + 1.353995860952899 x3_input_acceleration >= -2.445288886464569
 CBF_Constraint_obs1: - 0.44384048959543 x1_input_acceleration
   - 0.3119433008150119 x2_input_acceleration
   + 1.353995860952899 x3_input_acceleration >= -2.609509743256572
 CBF_Constraint_for_table: x3_input_acceleration >= -3.625004658447615
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6779082297775969 x3_input_acceleration + relaxation_CLF
   <= -20.18382382720008
 CBF_Constraint_obs2: - 0.1238414432697464 x1_input_acceleration
   - 0.3118846498445529 x2_input_acceleration
   + 1.356640623740408 x3_input_acceleration >= -2.391353149828197
 CBF_Constraint_obs3: 0.2561585567302536 x1_input_acceleration
   - 0.3118846498445529 x2_input_acceleration
   + 1.356640623740408 x3_input_acceleration >= -2.454203778721937
 CBF_Constraint_obs1: - 0.4438414432697464 x1_input_acceleration
   - 0.3118846498445529 x2_input_acceleration
   + 1.356640623740408 x3_input_acceleration >= -2.618426304443994
 CBF_Constraint_for_table: x3_input_acceleration >= -3.63121155185514
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6792305515667069 x3_input_acceleration + relaxation_CLF
   <= -20.18454165687265
 CBF_Constraint_obs2: - 0.1238414432697464 x1_input_acceleration
   - 0.3118846498445529 x2_input_acceleration
   + 1.359285267318628 x3_input_acceleration >= -2.400331469786585
 CBF_Constraint_obs3: 0.2561585567302536 x1_input_acceleration
   - 0.3118846498445529 x2_input_acceleration
   + 1.359285267318628 x3_input_acceleration >= -2.463182098680325
 CBF_Constraint_obs1: - 0.4438414432697464 x1_input_acceleration
   - 0.3118846498445529 x2_input_acceleration
   + 1.359285267318628 x3_input_acceleration >= -2.627404624402382
 CBF_Constraint_for_table: x3_input_acceleration >= -3.637776669177765
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6805528733558168 x3_input_acceleration + relaxation_CLF
   <= -20.18526088537314
 CBF_Constraint_obs2: - 0.1237804081134964 x1_input_acceleration
   - 0.3115308366731662 x2_input_acceleration
   + 1.362235921143125 x3_input_acceleration >= -2.410074695604606
 CBF_Constraint_obs3: 0.2562920711345504 x1_input_acceleration
   - 0.3114888750032443 x2_input_acceleration
   + 1.362242239235471 x3_input_acceleration >= -2.473018579168119
 CBF_Constraint_obs1: - 0.4437079288654496 x1_input_acceleration
   - 0.3114888750032443 x2_input_acceleration
   + 1.362242239235471 x3_input_acceleration >= -2.637007454682655
 CBF_Constraint_for_table: x3_input_acceleration >= -3.644388278123315
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6818751951449268 x3_input_acceleration + relaxation_CLF
   <= -20.18598151270157
 CBF_Constraint_obs2: - 0.1237804081134964 x1_input_acceleration
   - 0.3115308366731662 x2_input_acceleration
   + 1.364880564721345 x3_input_acceleration >= -2.419090009481355
 CBF_Constraint_obs3: 0.2562195918865036 x1_input_acceleration
   - 0.3115308366731662 x2_input_acceleration
   + 1.364880564721345 x3_input_acceleration >= -2.481998621773533
 CBF_Constraint_obs1: - 0.4437804081134964 x1_input_acceleration
   - 0.3115308366731662 x2_input_acceleration
   + 1.364880564721345 x3_input_acceleration >= -2.646114335972152
 CBF_Constraint_for_table: x3_input_acceleration >= -3.650953693469164
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6831975169340367 x3_input_acceleration + relaxation_CLF
   <= -20.18670353885793
 CBF_Constraint_obs2: - 0.1238657619648148 x1_input_acceleration
   - 0.3118928752855319 x2_input_acceleration
   + 1.367200839822697 x3_input_acceleration >= -2.427322536316111
 CBF_Constraint_obs3: 0.2561342380351852 x1_input_acceleration
   - 0.3118928752855319 x2_input_acceleration
   + 1.367200839822697 x3_input_acceleration >= -2.490150062449537
 CBF_Constraint_obs1: - 0.4438657619648148 x1_input_acceleration
   - 0.3118928752855319 x2_input_acceleration
   + 1.367200839822697 x3_input_acceleration >= -2.654415145887963
 CBF_Constraint_for_table: x3_input_acceleration >= -3.657611197991191
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6845198387231467 x3_input_acceleration + relaxation_CLF
   <= -20.18742696384222
 CBF_Constraint_obs2: - 0.1238657619648148 x1_input_acceleration
   - 0.3118928752855319 x2_input_acceleration
   + 1.369845483400917 x3_input_acceleration >= -2.436370676293616
 CBF_Constraint_obs3: 0.2561342380351852 x1_input_acceleration
   - 0.3118928752855319 x2_input_acceleration
   + 1.369845483400917 x3_input_acceleration >= -2.499198202427042
 CBF_Constraint_obs1: - 0.4438657619648148 x1_input_acceleration
   - 0.3118928752855319 x2_input_acceleration
   + 1.369845483400917 x3_input_acceleration >= -2.663463285865468
 CBF_Constraint_for_table: x3_input_acceleration >= -3.664222806936741
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6858421605122567 x3_input_acceleration + relaxation_CLF
   <= -20.18815178765445
 CBF_Constraint_obs2: - 0.1237808849506546 x1_input_acceleration
   - 0.3118575893358249 x2_input_acceleration
   + 1.372511942279125 x3_input_acceleration >= -2.445457369005418
 CBF_Constraint_obs3: 0.2561342380351852 x1_input_acceleration
   - 0.3118928752855319 x2_input_acceleration
   + 1.372490126979137 x3_input_acceleration >= -2.508263827753686
 CBF_Constraint_obs1: - 0.4438657619648148 x1_input_acceleration
   - 0.3118928752855319 x2_input_acceleration
   + 1.372490126979137 x3_input_acceleration >= -2.672528911192112
 CBF_Constraint_for_table: x3_input_acceleration >= -3.6708120641405
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6871644823013666 x3_input_acceleration + relaxation_CLF
   <= -20.1888780102946
 CBF_Constraint_obs2: - 0.1237727787189652 x1_input_acceleration
   - 0.3115346513704318 x2_input_acceleration
   + 1.375449006244613 x3_input_acceleration >= -2.455291885669504
 CBF_Constraint_obs3: 0.2562272212810348 x1_input_acceleration
   - 0.3115346513704318 x2_input_acceleration
   + 1.375449006244613 x3_input_acceleration >= -2.518207745886487
 CBF_Constraint_obs1: - 0.4437727787189652 x1_input_acceleration
   - 0.3115346513704318 x2_input_acceleration
   + 1.375449006244613 x3_input_acceleration >= -2.682310108644676
 CBF_Constraint_for_table: x3_input_acceleration >= -3.677441852502706
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6884868040904766 x3_input_acceleration + relaxation_CLF
   <= -20.18960563176269
 CBF_Constraint_obs2: - 0.1237751629047562 x1_input_acceleration
   - 0.3118298135713595 x2_input_acceleration
   + 1.377859761196734 x3_input_acceleration >= -2.46381955638646
 CBF_Constraint_obs3: 0.2562248370952438 x1_input_acceleration
   - 0.3118298135713595 x2_input_acceleration
   + 1.377859761196734 x3_input_acceleration >= -2.526733151626943
 CBF_Constraint_obs1: - 0.4437751629047562 x1_input_acceleration
   - 0.3118298135713595 x2_input_acceleration
   + 1.377859761196734 x3_input_acceleration >= -2.690839686710266
 CBF_Constraint_for_table: x3_input_acceleration >= -3.684045712844435
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6898091258795865 x3_input_acceleration + relaxation_CLF
   <= -20.1903346520587
 CBF_Constraint_obs2: - 0.1237985279255082 x1_input_acceleration
   - 0.3118388734773654 x2_input_acceleration
   + 1.380459224454214 x3_input_acceleration >= -2.472796536136107
 CBF_Constraint_obs3: 0.2562248370952438 x1_input_acceleration
   - 0.3118298135713595 x2_input_acceleration
   + 1.380504404774954 x3_input_acceleration >= -2.535851764224353
 CBF_Constraint_obs1: - 0.4437751629047562 x1_input_acceleration
   - 0.3118298135713595 x2_input_acceleration
   + 1.380504404774954 x3_input_acceleration >= -2.699958299307677
 CBF_Constraint_for_table: x3_input_acceleration >= -3.691107038834815
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6911314476686965 x3_input_acceleration + relaxation_CLF
   <= -20.19106507118265
 CBF_Constraint_obs2: - 0.1238648082904984 x1_input_acceleration
   - 0.3118841730073947 x2_input_acceleration
   + 1.383096357847192 x3_input_acceleration >= -2.481962203994397
 CBF_Constraint_obs3: 0.2562014720744918 x1_input_acceleration
   - 0.3118388734773654 x2_input_acceleration
   + 1.383103868032434 x3_input_acceleration >= -2.544823733838812
 CBF_Constraint_obs1: - 0.4437985279255082 x1_input_acceleration
   - 0.3118388734773654 x2_input_acceleration
   + 1.383103868032434 x3_input_acceleration >= -2.708971157708451
 CBF_Constraint_for_table: x3_input_acceleration >= -3.69730707770819
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6924537694578065 x3_input_acceleration + relaxation_CLF
   <= -20.19179688913453
 CBF_Constraint_obs2: - 0.1237827922992875 x1_input_acceleration
   - 0.311857827754404 x2_input_acceleration
   + 1.385733491240171 x3_input_acceleration >= -2.49104349143222
 CBF_Constraint_obs3: 0.2562172077007125 x1_input_acceleration
   - 0.311857827754404 x2_input_acceleration
   + 1.385733491240171 x3_input_acceleration >= -2.553949838747897
 CBF_Constraint_obs1: - 0.4438648082904984 x1_input_acceleration
   - 0.3118841730073947 x2_input_acceleration
   + 1.385741001425412 x3_input_acceleration >= -2.718207285553465
 CBF_Constraint_for_table: x3_input_acceleration >= -3.703865340496666
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6937760912469164 x3_input_acceleration + relaxation_CLF
   <= -20.19253010591434
 CBF_Constraint_obs2: - 0.1238457348041703 x1_input_acceleration
   - 0.311906107516672 x2_input_acceleration
   + 1.388347021193818 x3_input_acceleration >= -2.500163296491417
 CBF_Constraint_obs3: 0.2561542651958297 x1_input_acceleration
   - 0.311906107516672 x2_input_acceleration
   + 1.388347021193818 x3_input_acceleration >= -2.563009848427455
 CBF_Constraint_obs1: - 0.4438457348041703 x1_input_acceleration
   - 0.311906107516672 x2_input_acceleration
   + 1.388347021193818 x3_input_acceleration >= -2.727239884334753
 CBF_Constraint_for_table: x3_input_acceleration >= -3.710509135950395
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6950984130360264 x3_input_acceleration + relaxation_CLF
   <= -20.19326472152208
 CBF_Constraint_obs2: - 0.1238457348041703 x1_input_acceleration
   - 0.311906107516672 x2_input_acceleration
   + 1.390991664772038 x3_input_acceleration >= -2.509351246750839
 CBF_Constraint_obs3: 0.2561542651958297 x1_input_acceleration
   - 0.311906107516672 x2_input_acceleration
   + 1.390991664772038 x3_input_acceleration >= -2.572197798686878
 CBF_Constraint_obs1: - 0.4438457348041703 x1_input_acceleration
   - 0.311906107516672 x2_input_acceleration
   + 1.390991664772038 x3_input_acceleration >= -2.736427834594175
 CBF_Constraint_for_table: x3_input_acceleration >= -3.717120744895945
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6964207348251363 x3_input_acceleration + relaxation_CLF
   <= -20.19400073595775
 CBF_Constraint_obs2: - 0.1238595630817582 x1_input_acceleration
   - 0.3118831001237887 x2_input_acceleration
   + 1.39366766039341 x3_input_acceleration >= -2.518652258920086
 CBF_Constraint_obs3: 0.2561404369182418 x1_input_acceleration
   - 0.3118831001237887 x2_input_acceleration
   + 1.39366766039341 x3_input_acceleration >= -2.581485673992416
 CBF_Constraint_obs1: - 0.4438595630817582 x1_input_acceleration
   - 0.3118831001237887 x2_input_acceleration
   + 1.39366766039341 x3_input_acceleration >= -2.745739909385493
 CBF_Constraint_for_table: x3_input_acceleration >= -3.724165977632236
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6977430566142463 x3_input_acceleration + relaxation_CLF
   <= -20.19473814922136
 CBF_Constraint_obs2: - 0.1237751629047562 x1_input_acceleration
   - 0.311534055323984 x2_input_acceleration
   + 1.396581716966014 x3_input_acceleration >= -2.528517813428058
 CBF_Constraint_obs3: 0.2562248370952438 x1_input_acceleration
   - 0.311534055323984 x2_input_acceleration
   + 1.396581716966014 x3_input_acceleration >= -2.591431408668539
 CBF_Constraint_obs1: - 0.4437751629047562 x1_input_acceleration
   - 0.311534055323984 x2_input_acceleration
   + 1.396581716966014 x3_input_acceleration >= -2.755537943751863
 CBF_Constraint_for_table: x3_input_acceleration >= -3.730790699599636
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.6990653784033563 x3_input_acceleration + relaxation_CLF
   <= -20.19547696131289
 CBF_Constraint_obs2: - 0.1238657619648148 x1_input_acceleration
   - 0.3118869148210544 x2_input_acceleration
   + 1.398936801179915 x3_input_acceleration >= -2.537050435399017
 CBF_Constraint_obs3: 0.2561342380351852 x1_input_acceleration
   - 0.3118869148210544 x2_input_acceleration
   + 1.398936801179915 x3_input_acceleration >= -2.599877961532442
 CBF_Constraint_obs1: - 0.4438609935932328 x1_input_acceleration
   - 0.3119113527254123 x2_input_acceleration
   + 1.398928218111068 x3_input_acceleration >= -2.764126791274969
 CBF_Constraint_for_table: x3_input_acceleration >= -3.737337637505604
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7003877001924662 x3_input_acceleration + relaxation_CLF
   <= -20.19621717223236
 CBF_Constraint_obs2: - 0.1238347675495316 x1_input_acceleration
   - 0.3118867956117648 x2_input_acceleration
   + 1.401594200152117 x3_input_acceleration >= -2.546343406038843
 CBF_Constraint_obs3: 0.2561342380351852 x1_input_acceleration
   - 0.3118869148210544 x2_input_acceleration
   + 1.401581444758135 x3_input_acceleration >= -2.609135927275952
 CBF_Constraint_obs1: - 0.4438657619648148 x1_input_acceleration
   - 0.3118869148210544 x2_input_acceleration
   + 1.401581444758135 x3_input_acceleration >= -2.773401010714379
 CBF_Constraint_for_table: x3_input_acceleration >= -3.743982625052229
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7017100219815762 x3_input_acceleration + relaxation_CLF
   <= -20.19695878197975
 CBF_Constraint_obs2: - 0.1238347675495316 x1_input_acceleration
   - 0.3118867956117648 x2_input_acceleration
   + 1.404238843730337 x3_input_acceleration >= -2.555618941465169
 CBF_Constraint_obs3: 0.2561652324504684 x1_input_acceleration
   - 0.3118867956117648 x2_input_acceleration
   + 1.404238843730337 x3_input_acceleration >= -2.618475912293114
 CBF_Constraint_obs1: - 0.4438347675495316 x1_input_acceleration
   - 0.3118867956117648 x2_input_acceleration
   + 1.404238843730337 x3_input_acceleration >= -2.782686755504794
 CBF_Constraint_for_table: x3_input_acceleration >= -3.750594233997778
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7030323437706861 x3_input_acceleration + relaxation_CLF
   <= -20.19770179055508
 CBF_Constraint_obs2: - 0.1238409664325882 x1_input_acceleration
   - 0.3118959747270602 x2_input_acceleration
   + 1.406891712749536 x3_input_acceleration >= -2.564949969276529
 CBF_Constraint_obs3: 0.2561652324504684 x1_input_acceleration
   - 0.3118867956117648 x2_input_acceleration
   + 1.406883487308557 x3_input_acceleration >= -2.627768933068579
 CBF_Constraint_obs1: - 0.4438347675495316 x1_input_acceleration
   - 0.3118867956117648 x2_input_acceleration
   + 1.406883487308557 x3_input_acceleration >= -2.79197977628026
 CBF_Constraint_for_table: x3_input_acceleration >= -3.756745397062438
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7043546655597961 x3_input_acceleration + relaxation_CLF
   <= -20.19844619795834
 CBF_Constraint_obs2: - 0.1238505031757523 x1_input_acceleration
   - 0.311882504077341 x2_input_acceleration
   + 1.409529084561093 x3_input_acceleration >= -2.574227354637592
 CBF_Constraint_obs3: 0.2561590335674118 x1_input_acceleration
   - 0.3118959747270602 x2_input_acceleration
   + 1.409536356327756 x3_input_acceleration >= -2.637111611673574
 CBF_Constraint_obs1: - 0.4438409664325882 x1_input_acceleration
   - 0.3118959747270602 x2_input_acceleration
   + 1.409536356327756 x3_input_acceleration >= -2.801333302930604
 CBF_Constraint_for_table: x3_input_acceleration >= -3.763368032867271
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.705676987348906 x3_input_acceleration + relaxation_CLF
   <= -20.19919200418953
 CBF_Constraint_obs2: - 0.1237513210468461 x1_input_acceleration
   - 0.3115946136430758 x2_input_acceleration
   + 1.412474612386139 x3_input_acceleration >= -2.58436265418449
 CBF_Constraint_obs3: 0.256248678953154 x1_input_acceleration
   - 0.3115946136430758 x2_input_acceleration
   + 1.412474612386139 x3_input_acceleration >= -2.647298899189986
 CBF_Constraint_obs1: - 0.4437513210468461 x1_input_acceleration
   - 0.3115946136430758 x2_input_acceleration
   + 1.412474612386139 x3_input_acceleration >= -2.811363711021967
 CBF_Constraint_for_table: x3_input_acceleration >= -3.770010338204881
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.706999309138016 x3_input_acceleration + relaxation_CLF
   <= -20.19993920924865
 CBF_Constraint_obs2: - 0.1236902858905961 x1_input_acceleration
   - 0.3114950738863009 x2_input_acceleration
   + 1.415133203451237 x3_input_acceleration >= -2.593663064236611
 CBF_Constraint_obs3: 0.256309714109404 x1_input_acceleration
   - 0.3114950738863009 x2_input_acceleration
   + 1.415133203451237 x3_input_acceleration >= -2.656657292640545
 CBF_Constraint_obs1: - 0.4436902858905961 x1_input_acceleration
   - 0.3114950738863009 x2_input_acceleration
   + 1.415133203451237 x3_input_acceleration >= -2.820615292949088
 CBF_Constraint_for_table: x3_input_acceleration >= -3.77658886657258
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.708321630927126 x3_input_acceleration + relaxation_CLF
   <= -20.2006878131357
 CBF_Constraint_obs2: - 0.1238538410358597 x1_input_acceleration
   - 0.311883815379526 x2_input_acceleration
   + 1.417450855948219 x3_input_acceleration >= -2.602222771579206
 CBF_Constraint_obs3: 0.2561461589641403 x1_input_acceleration
   - 0.311883815379526 x2_input_acceleration
   + 1.417450855948219 x3_input_acceleration >= -2.665061622595139
 CBF_Constraint_obs1: - 0.4438538410358597 x1_input_acceleration
   - 0.311883815379526 x2_input_acceleration
   + 1.417450855948219 x3_input_acceleration >= -2.829305844407894
 CBF_Constraint_for_table: x3_input_acceleration >= -3.783196899239443
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7096439527162359 x3_input_acceleration + relaxation_CLF
   <= -20.20143781585069
 CBF_Constraint_obs2: - 0.1237227108173539 x1_input_acceleration
   - 0.3114972196535128 x2_input_acceleration
   + 1.420409377585826 x3_input_acceleration >= -2.612375783697021
 CBF_Constraint_obs3: 0.2562772891826461 x1_input_acceleration
   - 0.3114972196535128 x2_input_acceleration
   + 1.420409377585826 x3_input_acceleration >= -2.675339208420535
 CBF_Constraint_obs1: - 0.4437227108173539 x1_input_acceleration
   - 0.3114972196535128 x2_input_acceleration
   + 1.420409377585826 x3_input_acceleration >= -2.839353952350904
 CBF_Constraint_for_table: x3_input_acceleration >= -3.789828475740993
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7109662745053459 x3_input_acceleration + relaxation_CLF
   <= -20.20218921739361
 CBF_Constraint_obs2: - 0.1236955310993363 x1_input_acceleration
   - 0.3114944778398532 x2_input_acceleration
   + 1.423083346649275 x3_input_acceleration >= -2.621869507061464
 CBF_Constraint_obs3: 0.2563044689006637 x1_input_acceleration
   - 0.3114944778398532 x2_input_acceleration
   + 1.423083346649275 x3_input_acceleration >= -2.684858752517095
 CBF_Constraint_obs1: - 0.4436955310993363 x1_input_acceleration
   - 0.3114944778398532 x2_input_acceleration
   + 1.423083346649275 x3_input_acceleration >= -2.848825931940933
 CBF_Constraint_for_table: x3_input_acceleration >= -3.796470185032154
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7122885962944558 x3_input_acceleration + relaxation_CLF
   <= -20.20294201776445
 CBF_Constraint_obs2: - 0.1237713482074906 x1_input_acceleration
   - 0.3118335090593356 x2_input_acceleration
   + 1.425447490769182 x3_input_acceleration >= -2.630575041186896
 CBF_Constraint_obs3: 0.2562286517925094 x1_input_acceleration
   - 0.3118335090593356 x2_input_acceleration
   + 1.425447490769182 x3_input_acceleration >= -2.69349226038978
 CBF_Constraint_obs1: - 0.4437713482074906 x1_input_acceleration
   - 0.3118335090593356 x2_input_acceleration
   + 1.425447490769182 x3_input_acceleration >= -2.857592119752888
 CBF_Constraint_for_table: x3_input_acceleration >= -3.803108616067853
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7136109180835658 x3_input_acceleration + relaxation_CLF
   <= -20.20369621696323
 CBF_Constraint_obs2: - 0.123769917696016 x1_input_acceleration
   - 0.3118553243593234 x2_input_acceleration
   + 1.428070199838125 x3_input_acceleration >= -2.639946539404299
 CBF_Constraint_obs3: 0.256230082303984 x1_input_acceleration
   - 0.3118553243593234 x2_input_acceleration
   + 1.428070199838125 x3_input_acceleration >= -2.702865117593084
 CBF_Constraint_obs1: - 0.4437713482074906 x1_input_acceleration
   - 0.3118335090593356 x2_input_acceleration
   + 1.428092134347402 x3_input_acceleration >= -2.867025363808839
 CBF_Constraint_for_table: x3_input_acceleration >= -3.809613532699255
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7149332398726758 x3_input_acceleration + relaxation_CLF
   <= -20.20445181498994
 CBF_Constraint_obs2: - 0.1237775470905472 x1_input_acceleration
   - 0.3115493141130465 x2_input_acceleration
   + 1.43096816315664 x3_input_acceleration >= -2.650067175319939
 CBF_Constraint_obs3: 0.2562224529094528 x1_input_acceleration
   - 0.3115493141130465 x2_input_acceleration
   + 1.43096816315664 x3_input_acceleration >= -2.712978505583919
 CBF_Constraint_obs1: - 0.4437775470905472 x1_input_acceleration
   - 0.3115493141130465 x2_input_acceleration
   + 1.43096816315664 x3_input_acceleration >= -2.877089212992377
 CBF_Constraint_for_table: x3_input_acceleration >= -3.816245407224028
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7162555616617857 x3_input_acceleration + relaxation_CLF
   <= -20.20520881184457
 CBF_Constraint_obs2: - 0.1236950542621781 x1_input_acceleration
   - 0.3114895902589816 x2_input_acceleration
   + 1.433646781382382 x3_input_acceleration >= -2.659586656320869
 CBF_Constraint_obs3: 0.2563049457378219 x1_input_acceleration
   - 0.3114895902589816 x2_input_acceleration
   + 1.433646781382382 x3_input_acceleration >= -2.7225763547718
 CBF_Constraint_obs1: - 0.4436950542621781 x1_input_acceleration
   - 0.3114895902589816 x2_input_acceleration
   + 1.433646781382382 x3_input_acceleration >= -2.886542699730612
 CBF_Constraint_for_table: x3_input_acceleration >= -3.82336991413787
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7175778834508957 x3_input_acceleration + relaxation_CLF
   <= -20.20596720752715
 CBF_Constraint_obs2: - 0.1237141277485062 x1_input_acceleration
   - 0.3114914976076144 x2_input_acceleration
   + 1.436288206309784 x3_input_acceleration >= -2.669069937588731
 CBF_Constraint_obs3: 0.2562858722514938 x1_input_acceleration
   - 0.3114914976076144 x2_input_acceleration
   + 1.436288206309784 x3_input_acceleration >= -2.732041516227651
 CBF_Constraint_obs1: - 0.4437141277485062 x1_input_acceleration
   - 0.3114914976076144 x2_input_acceleration
   + 1.436288206309784 x3_input_acceleration >= -2.896041239787536
 CBF_Constraint_for_table: x3_input_acceleration >= -3.829499619530411
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7189002052400056 x3_input_acceleration + relaxation_CLF
   <= -20.20672700203765
 CBF_Constraint_obs2: - 0.1237780239277054 x1_input_acceleration
   - 0.3115426383928317 x2_input_acceleration
   + 1.43891544533173 x3_input_acceleration >= -2.67857184194803
 CBF_Constraint_obs3: 0.2562219760722946 x1_input_acceleration
   - 0.3115426383928317 x2_input_acceleration
   + 1.43891544533173 x3_input_acceleration >= -2.74148271921671
 CBF_Constraint_obs1: - 0.4437141277485062 x1_input_acceleration
   - 0.3114914976076144 x2_input_acceleration
   + 1.438932849888004 x3_input_acceleration >= -2.905546158415332
 CBF_Constraint_for_table: x3_input_acceleration >= -3.836148183355721
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7202225270291156 x3_input_acceleration + relaxation_CLF
   <= -20.20748819537608
 CBF_Constraint_obs2: - 0.1237823154621293 x1_input_acceleration
   - 0.3115344129518527 x2_input_acceleration
   + 1.441554009236182 x3_input_acceleration >= -2.688067142024166
 CBF_Constraint_obs3: 0.2562176845378707 x1_input_acceleration
   - 0.3115344129518527 x2_input_acceleration
   + 1.441554009236182 x3_input_acceleration >= -2.750973942335144
 CBF_Constraint_obs1: - 0.4437823154621293 x1_input_acceleration
   - 0.3115344129518527 x2_input_acceleration
   + 1.441554009236182 x3_input_acceleration >= -2.91509299439387
 CBF_Constraint_for_table: x3_input_acceleration >= -3.842706148120973
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7215448488182256 x3_input_acceleration + relaxation_CLF
   <= -20.20825078754245
 CBF_Constraint_obs2: - 0.1238409664325882 x1_input_acceleration
   - 0.3119114719347018 x2_input_acceleration
   + 1.443903848241343 x3_input_acceleration >= -2.696854592821987
 CBF_Constraint_obs3: 0.2561590335674118 x1_input_acceleration
   - 0.3119114719347018 x2_input_acceleration
   + 1.443903848241343 x3_input_acceleration >= -2.759705674711028
 CBF_Constraint_obs1: - 0.4438409664325882 x1_input_acceleration
   - 0.3119114719347018 x2_input_acceleration
   + 1.443903848241343 x3_input_acceleration >= -2.923927365968058
 CBF_Constraint_for_table: x3_input_acceleration >= -3.84938511031512
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7228671706073355 x3_input_acceleration + relaxation_CLF
   <= -20.20901477853674
 CBF_Constraint_obs2: - 0.1238409664325882 x1_input_acceleration
   - 0.3119114719347018 x2_input_acceleration
   + 1.446548491819563 x3_input_acceleration >= -2.706409863096103
 CBF_Constraint_obs3: 0.2561590335674118 x1_input_acceleration
   - 0.3119114719347018 x2_input_acceleration
   + 1.446548491819563 x3_input_acceleration >= -2.769260944985144
 CBF_Constraint_obs1: - 0.4438409664325882 x1_input_acceleration
   - 0.3119114719347018 x2_input_acceleration
   + 1.446548491819563 x3_input_acceleration >= -2.933482636242174
 CBF_Constraint_for_table: x3_input_acceleration >= -3.855996719260669
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7241894923964455 x3_input_acceleration + relaxation_CLF
   <= -20.20978016835897
 CBF_Constraint_obs2: - 0.1238414432697464 x1_input_acceleration
   - 0.3118819080308932 x2_input_acceleration
   + 1.449205533163896 x3_input_acceleration >= -2.716004631226222
 CBF_Constraint_obs3: 0.2561585567302536 x1_input_acceleration
   - 0.3118819080308932 x2_input_acceleration
   + 1.449205533163896 x3_input_acceleration >= -2.778855260119963
 CBF_Constraint_obs1: - 0.4438414432697464 x1_input_acceleration
   - 0.3118819080308932 x2_input_acceleration
   + 1.449205533163896 x3_input_acceleration >= -2.943077785842019
 CBF_Constraint_for_table: x3_input_acceleration >= -3.86260534797398
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7255118141855554 x3_input_acceleration + relaxation_CLF
   <= -20.21054695700913
 CBF_Constraint_obs2: - 0.1238023426227738 x1_input_acceleration
   - 0.3115224920228976 x2_input_acceleration
   + 1.452119470527212 x3_input_acceleration >= -2.72628029969909
 CBF_Constraint_obs3: 0.2561976573772262 x1_input_acceleration
   - 0.3115224920228976 x2_input_acceleration
   + 1.452119470527212 x3_input_acceleration >= -2.789168074207455
 CBF_Constraint_obs1: - 0.4438023426227738 x1_input_acceleration
   - 0.3115224920228976 x2_input_acceleration
   + 1.452119470527212 x3_input_acceleration >= -2.953322173797309
 CBF_Constraint_for_table: x3_input_acceleration >= -3.86921695691953
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7268341359746654 x3_input_acceleration + relaxation_CLF
   <= -20.21131514448722
 CBF_Constraint_obs2: - 0.1237737323932816 x1_input_acceleration
   - 0.3115403734163302 x2_input_acceleration
   + 1.45477257796499 x3_input_acceleration >= -2.735925743372315
 CBF_Constraint_obs3: 0.2562262676067184 x1_input_acceleration
   - 0.3115403734163302 x2_input_acceleration
   + 1.45477257796499 x3_input_acceleration >= -2.798840697598697
 CBF_Constraint_obs1: - 0.4437737323932816 x1_input_acceleration
   - 0.3115403734163302 x2_input_acceleration
   + 1.45477257796499 x3_input_acceleration >= -2.96294472928694
 CBF_Constraint_for_table: x3_input_acceleration >= -3.876228513031523
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7281564577637754 x3_input_acceleration + relaxation_CLF
   <= -20.21208473079324
 CBF_Constraint_obs2: - 0.123807587831514 x1_input_acceleration
   - 0.311862596125986 x2_input_acceleration
   + 1.457144828316586 x3_input_acceleration >= -2.744822060446088
 CBF_Constraint_obs3: 0.256192412168486 x1_input_acceleration
   - 0.311862596125986 x2_input_acceleration
   + 1.457144828316586 x3_input_acceleration >= -2.80770485200615
 CBF_Constraint_obs1: - 0.443807587831514 x1_input_acceleration
   - 0.311862596125986 x2_input_acceleration
   + 1.457144828316586 x3_input_acceleration >= -2.971868130711299
 CBF_Constraint_for_table: x3_input_acceleration >= -3.882451797716361
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7294787795528853 x3_input_acceleration + relaxation_CLF
   <= -20.21285571592719
 CBF_Constraint_obs2: - 0.1237627651386429 x1_input_acceleration
   - 0.3115332208589572 x2_input_acceleration
   + 1.460087137490814 x3_input_acceleration >= -2.755280773499548
 CBF_Constraint_obs3: 0.2562372348613571 x1_input_acceleration
   - 0.3115332208589572 x2_input_acceleration
   + 1.460087137490814 x3_input_acceleration >= -2.818206146617838
 CBF_Constraint_obs1: - 0.4436812259845902 x1_input_acceleration
   - 0.3114892326311129 x2_input_acceleration
   + 1.460111575395172 x3_input_acceleration >= -2.9822554811741
 CBF_Constraint_for_table: x3_input_acceleration >= -3.889062512592239
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7308011013419953 x3_input_acceleration + relaxation_CLF
   <= -20.21362809988907
 CBF_Constraint_obs2: - 0.1238500263385941 x1_input_acceleration
   - 0.3118918024019259 x2_input_acceleration
   + 1.462409796777957 x3_input_acceleration >= -2.764072173927174
 CBF_Constraint_obs3: 0.2562372348613571 x1_input_acceleration
   - 0.3115332208589572 x2_input_acceleration
   + 1.462731781069034 x3_input_acceleration >= -2.827868414471923
 CBF_Constraint_obs1: - 0.4437627651386429 x1_input_acceleration
   - 0.3115332208589572 x2_input_acceleration
   + 1.462731781069034 x3_input_acceleration >= -2.991953253464549
 CBF_Constraint_for_table: x3_input_acceleration >= -3.895651471772774
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7321234231311052 x3_input_acceleration + relaxation_CLF
   <= -20.21440188267888
 CBF_Constraint_obs2: - 0.1237899448566605 x1_input_acceleration
   - 0.3118482910112399 x2_input_acceleration
   + 1.465076970911902 x3_input_acceleration >= -2.773779797188301
 CBF_Constraint_obs3: 0.2562100551433395 x1_input_acceleration
   - 0.3118482910112399 x2_input_acceleration
   + 1.465076970911902 x3_input_acceleration >= -2.836679349574474
 CBF_Constraint_obs1: - 0.4437899448566605 x1_input_acceleration
   - 0.3118482910112399 x2_input_acceleration
   + 1.465076970911902 x3_input_acceleration >= -3.00081175307363
 CBF_Constraint_for_table: x3_input_acceleration >= -3.90226814711313
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7334457449202152 x3_input_acceleration + relaxation_CLF
   <= -20.21517706429663
 CBF_Constraint_obs2: - 0.1237069751911332 x1_input_acceleration
   - 0.311482318492319 x2_input_acceleration
   + 1.468021425853343 x3_input_acceleration >= -2.784264446510964
 CBF_Constraint_obs3: 0.2562100551433395 x1_input_acceleration
   - 0.3118482910112399 x2_input_acceleration
   + 1.467721614490122 x3_input_acceleration >= -2.846374608255844
 CBF_Constraint_obs1: - 0.4437899448566605 x1_input_acceleration
   - 0.3118482910112399 x2_input_acceleration
   + 1.467721614490122 x3_input_acceleration >= -3.010507011755
 CBF_Constraint_for_table: x3_input_acceleration >= -3.908855318154322
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7347680667093252 x3_input_acceleration + relaxation_CLF
   <= -20.2159536447423
 CBF_Constraint_obs2: - 0.1237069751911332 x1_input_acceleration
   - 0.311482318492319 x2_input_acceleration
   + 1.470666069431563 x3_input_acceleration >= -2.793979172776964
 CBF_Constraint_obs3: 0.2562930248088668 x1_input_acceleration
   - 0.311482318492319 x2_input_acceleration
   + 1.470666069431563 x3_input_acceleration >= -2.856957546345388
 CBF_Constraint_obs1: - 0.4437069751911332 x1_input_acceleration
   - 0.311482318492319 x2_input_acceleration
   + 1.470666069431563 x3_input_acceleration >= -3.020944752929871
 CBF_Constraint_for_table: x3_input_acceleration >= -3.915466927099871
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7360903884984351 x3_input_acceleration + relaxation_CLF
   <= -20.21673162401591
 CBF_Constraint_obs2: - 0.1237837459736039 x1_input_acceleration
   - 0.3115482412294406 x2_input_acceleration
   + 1.473270181851335 x3_input_acceleration >= -2.803637188890852
 CBF_Constraint_obs3: 0.2562162540263961 x1_input_acceleration
   - 0.3115482412294406 x2_input_acceleration
   + 1.473270181851335 x3_input_acceleration >= -2.866542630215929
 CBF_Constraint_obs1: - 0.4437069751911332 x1_input_acceleration
   - 0.311482318492319 x2_input_acceleration
   + 1.473310713009782 x3_input_acceleration >= -3.030676964545011
 CBF_Constraint_for_table: x3_input_acceleration >= -3.922508583557476
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7374127102875451 x3_input_acceleration + relaxation_CLF
   <= -20.21751100211745
 CBF_Constraint_obs2: - 0.1237613346271683 x1_input_acceleration
   - 0.3118475757555026 x2_input_acceleration
   + 1.475693811406728 x3_input_acceleration >= -2.812797504346747
 CBF_Constraint_obs3: 0.2562386653728317 x1_input_acceleration
   - 0.3118475757555026 x2_input_acceleration
   + 1.475693811406728 x3_input_acceleration >= -2.875724236450937
 CBF_Constraint_obs1: - 0.4437613346271683 x1_input_acceleration
   - 0.3118475757555026 x2_input_acceleration
   + 1.475693811406728 x3_input_acceleration >= -3.039806572048481
 CBF_Constraint_for_table: x3_input_acceleration >= -3.928618619417241
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.738735032076655 x3_input_acceleration + relaxation_CLF
   <= -20.21829177904692
 CBF_Constraint_obs2: - 0.1236941005878617 x1_input_acceleration
   - 0.3114955507234591 x2_input_acceleration
   + 1.478605006956383 x3_input_acceleration >= -2.823253094046547
 CBF_Constraint_obs3: 0.2563058994121383 x1_input_acceleration
   - 0.3114955507234591 x2_input_acceleration
   + 1.478605006956383 x3_input_acceleration >= -2.886243698488078
 CBF_Constraint_obs1: - 0.4436941005878617 x1_input_acceleration
   - 0.3114955507234591 x2_input_acceleration
   + 1.478605006956383 x3_input_acceleration >= -3.050208374516836
 CBF_Constraint_for_table: x3_input_acceleration >= -3.935267481265775
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.740057353865765 x3_input_acceleration + relaxation_CLF
   <= -20.21907395480432
 CBF_Constraint_obs2: - 0.1238352443866898 x1_input_acceleration
   - 0.3119412742570895 x2_input_acceleration
   + 1.480957826193783 x3_input_acceleration >= -2.832348261627579
 CBF_Constraint_obs3: 0.2562386653728317 x1_input_acceleration
   - 0.3118434034303683 x2_input_acceleration
   + 1.480984648283932 x3_input_acceleration >= -2.895275112934709
 CBF_Constraint_obs1: - 0.4437613346271683 x1_input_acceleration
   - 0.3118434034303683 x2_input_acceleration
   + 1.480984648283932 x3_input_acceleration >= -3.059357448532253
 CBF_Constraint_for_table: x3_input_acceleration >= -3.942350562951498
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7413796756548749 x3_input_acceleration + relaxation_CLF
   <= -20.21985752938965
 CBF_Constraint_obs2: - 0.1238352443866898 x1_input_acceleration
   - 0.3119412742570895 x2_input_acceleration
   + 1.483602469772003 x3_input_acceleration >= -2.842148518313794
 CBF_Constraint_obs3: 0.2561647556133102 x1_input_acceleration
   - 0.3119412742570895 x2_input_acceleration
   + 1.483602469772003 x3_input_acceleration >= -2.905005036146438
 CBF_Constraint_obs1: - 0.4438352443866898 x1_input_acceleration
   - 0.3119412742570895 x2_input_acceleration
   + 1.483602469772003 x3_input_acceleration >= -3.069216713823145
 CBF_Constraint_for_table: x3_input_acceleration >= -3.94855030380165
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7427019974439849 x3_input_acceleration + relaxation_CLF
   <= -20.22064250280291
 CBF_Constraint_obs2: - 0.1237856533222367 x1_input_acceleration
   - 0.3115140281633395 x2_input_acceleration
   + 1.48651056588013 x3_input_acceleration >= -2.852596925230284
 CBF_Constraint_obs3: 0.2561647556133102 x1_input_acceleration
   - 0.3119412742570895 x2_input_acceleration
   + 1.486247113350223 x3_input_acceleration >= -2.914822778181792
 CBF_Constraint_obs1: - 0.4438352443866898 x1_input_acceleration
   - 0.3119412742570895 x2_input_acceleration
   + 1.486247113350223 x3_input_acceleration >= -3.079034455858499
 CBF_Constraint_for_table: x3_input_acceleration >= -3.955585105725105
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7440243192330949 x3_input_acceleration + relaxation_CLF
   <= -20.22142887504411
 CBF_Constraint_obs2: - 0.1237241413288285 x1_input_acceleration
   - 0.3114686094240207 x2_input_acceleration
   + 1.489178693688392 x3_input_acceleration >= -2.862466924424441
 CBF_Constraint_obs3: 0.2562758586711715 x1_input_acceleration
   - 0.3114686094240207 x2_input_acceleration
   + 1.489178693688392 x3_input_acceleration >= -2.925428990162055
 CBF_Constraint_obs1: - 0.4437241413288285 x1_input_acceleration
   - 0.3114686094240207 x2_input_acceleration
   + 1.489178693688392 x3_input_acceleration >= -3.089446237487504
 CBF_Constraint_for_table: x3_input_acceleration >= -3.961712426931855
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7453466410222048 x3_input_acceleration + relaxation_CLF
   <= -20.22221664611323
 CBF_Constraint_obs2: - 0.1238495495014359 x1_input_acceleration
   - 0.311879047007944 x2_input_acceleration
   + 1.491521737764048 x3_input_acceleration >= -2.871555431371222
 CBF_Constraint_obs3: 0.2561504504985641 x1_input_acceleration
   - 0.311879047007944 x2_input_acceleration
   + 1.491521737764048 x3_input_acceleration >= -2.934398359344858
 CBF_Constraint_obs1: - 0.4438495495014359 x1_input_acceleration
   - 0.311879047007944 x2_input_acceleration
   + 1.491521737764048 x3_input_acceleration >= -3.098635070972371
 CBF_Constraint_for_table: x3_input_acceleration >= -3.968367249244867
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7466689628113148 x3_input_acceleration + relaxation_CLF
   <= -20.22300581601029
 CBF_Constraint_obs2: - 0.123708882539766 x1_input_acceleration
   - 0.3114882789567965 x2_input_acceleration
   + 1.494481809122419 x3_input_acceleration >= -2.882255891680655
 CBF_Constraint_obs3: 0.256291117460234 x1_input_acceleration
   - 0.3114882789567965 x2_input_acceleration
   + 1.494481809122419 x3_input_acceleration >= -2.945232453267877
 CBF_Constraint_obs1: - 0.4438495495014359 x1_input_acceleration
   - 0.311879047007944 x2_input_acceleration
   + 1.494166381342268 x3_input_acceleration >= -3.108505172110823
 CBF_Constraint_for_table: x3_input_acceleration >= -3.975370162683367
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7479912846004247 x3_input_acceleration + relaxation_CLF
   <= -20.22379638473528
 CBF_Constraint_obs2: - 0.1234990741901566 x1_input_acceleration
   - 0.3117894016222018 x2_input_acceleration
   + 1.496937267558122 x3_input_acceleration >= -2.891607294117667
 CBF_Constraint_obs3: 0.2565009258098434 x1_input_acceleration
   - 0.3117894016222018 x2_input_acceleration
   + 1.496937267558122 x3_input_acceleration >= -2.954783173637018
 CBF_Constraint_obs1: - 0.4434990741901566 x1_input_acceleration
   - 0.3117894016222018 x2_input_acceleration
   + 1.496937267558122 x3_input_acceleration >= -3.118406553469792
 CBF_Constraint_for_table: x3_input_acceleration >= -3.981588380973399
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7493136063895347 x3_input_acceleration + relaxation_CLF
   <= -20.2245883522882
 CBF_Constraint_obs2: - 0.1238452579670121 x1_input_acceleration
   - 0.3118895374254245 x2_input_acceleration
   + 1.499438502361012 x3_input_acceleration >= -2.901160692298746
 CBF_Constraint_obs3: 0.2561547420329879 x1_input_acceleration
   - 0.3118895374254245 x2_input_acceleration
   + 1.499438502361012 x3_input_acceleration >= -2.964007697230084
 CBF_Constraint_obs1: - 0.4438452579670121 x1_input_acceleration
   - 0.3118895374254245 x2_input_acceleration
   + 1.499438502361012 x3_input_acceleration >= -3.128236898672355
 CBF_Constraint_for_table: x3_input_acceleration >= -3.988645832661869
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7506359281786447 x3_input_acceleration + relaxation_CLF
   <= -20.22538171866904
 CBF_Constraint_obs2: - 0.1237060215168168 x1_input_acceleration
   - 0.3114724241212863 x2_input_acceleration
   + 1.502424442135216 x3_input_acceleration >= -2.911996818841032
 CBF_Constraint_obs3: 0.2562939784831832 x1_input_acceleration
   - 0.3114724241212863 x2_input_acceleration
   + 1.502424442135216 x3_input_acceleration >= -2.974976098400056
 CBF_Constraint_obs1: - 0.4438452579670121 x1_input_acceleration
   - 0.3118895374254245 x2_input_acceleration
   + 1.502083145939232 x3_input_acceleration >= -3.138159342362437
 CBF_Constraint_for_table: x3_input_acceleration >= -3.99481249293417
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7519582499677546 x3_input_acceleration + relaxation_CLF
   <= -20.22617648387783
 CBF_Constraint_obs2: - 0.1238280918293168 x1_input_acceleration
   - 0.3119454465822238 x2_input_acceleration
   + 1.504731961842587 x3_input_acceleration >= -2.92107704370014
 CBF_Constraint_obs3: 0.2562524936504196 x1_input_acceleration
   - 0.3115829311326999 x2_input_acceleration
   + 1.505031653996517 x3_input_acceleration >= -2.984836928758494
 CBF_Constraint_obs1: - 0.4437475063495804 x1_input_acceleration
   - 0.3115829311326999 x2_input_acceleration
   + 1.505031653996517 x3_input_acceleration >= -3.14889506487026
 CBF_Constraint_for_table: x3_input_acceleration >= -4.001406518509512
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7532805717568646 x3_input_acceleration + relaxation_CLF
   <= -20.22697264791454
 CBF_Constraint_obs2: - 0.1238280918293168 x1_input_acceleration
   - 0.3119454465822238 x2_input_acceleration
   + 1.507376605420806 x3_input_acceleration >= -2.931034485674282
 CBF_Constraint_obs3: 0.2561719081706832 x1_input_acceleration
   - 0.3119454465822238 x2_input_acceleration
   + 1.507376605420806 x3_input_acceleration >= -2.993897798436432
 CBF_Constraint_obs1: - 0.4438280918293168 x1_input_acceleration
   - 0.3119454465822238 x2_input_acceleration
   + 1.507376605420806 x3_input_acceleration >= -3.158096959137736
 CBF_Constraint_for_table: x3_input_acceleration >= -4.008018127455061
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7546028935459745 x3_input_acceleration + relaxation_CLF
   <= -20.22777021077918
 CBF_Constraint_obs2: - 0.1237732555561234 x1_input_acceleration
   - 0.3115369163469333 x2_input_acceleration
   + 1.51031748408356 x3_input_acceleration >= -2.941792464708015
 CBF_Constraint_obs3: 0.2562267444438766 x1_input_acceleration
   - 0.3115369163469333 x2_input_acceleration
   + 1.51031748408356 x3_input_acceleration >= -3.004707871929698
 CBF_Constraint_obs1: - 0.4437732555561234 x1_input_acceleration
   - 0.3115369163469333 x2_input_acceleration
   + 1.51031748408356 x3_input_acceleration >= -3.168811069152913
 CBF_Constraint_for_table: x3_input_acceleration >= -4.014708116508491
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7559252153350845 x3_input_acceleration + relaxation_CLF
   <= -20.22856917247176
 CBF_Constraint_obs2: - 0.1238485958271195 x1_input_acceleration
   - 0.3118810735658664 x2_input_acceleration
   + 1.512690449690894 x3_input_acceleration >= -2.951050844153731
 CBF_Constraint_obs3: 0.2561514041728805 x1_input_acceleration
   - 0.3118810735658664 x2_input_acceleration
   + 1.512690449690894 x3_input_acceleration >= -3.013894678117968
 CBF_Constraint_obs1: - 0.4437732555561234 x1_input_acceleration
   - 0.3115369163469333 x2_input_acceleration
   + 1.51296212766178 x3_input_acceleration >= -3.178805440415871
 CBF_Constraint_for_table: x3_input_acceleration >= -4.021693744600007
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7572475371241945 x3_input_acceleration + relaxation_CLF
   <= -20.22936953299226
 CBF_Constraint_obs2: - 0.1237789776020218 x1_input_acceleration
   - 0.3118337474779147 x2_input_acceleration
   + 1.515357146987681 x3_input_acceleration >= -2.961086005361206
 CBF_Constraint_obs3: 0.2562210223979782 x1_input_acceleration
   - 0.3118337474779147 x2_input_acceleration
   + 1.515357146987681 x3_input_acceleration >= -3.023995976639285
 CBF_Constraint_obs1: - 0.4437789776020218 x1_input_acceleration
   - 0.3118337474779147 x2_input_acceleration
   + 1.515357146987681 x3_input_acceleration >= -3.188109187442823
 CBF_Constraint_for_table: x3_input_acceleration >= -4.027857126616845
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7585698589133044 x3_input_acceleration + relaxation_CLF
   <= -20.2301712923407
 CBF_Constraint_obs2: - 0.1238476421528031 x1_input_acceleration
   - 0.3118808351472873 x2_input_acceleration
   + 1.517956610245161 x3_input_acceleration >= -2.971000205482446
 CBF_Constraint_obs3: 0.2561523578471969 x1_input_acceleration
   - 0.3118808351472873 x2_input_acceleration
   + 1.517956610245161 x3_input_acceleration >= -3.033844945437283
 CBF_Constraint_obs1: - 0.4438476421528031 x1_input_acceleration
   - 0.3118808351472873 x2_input_acceleration
   + 1.517956610245161 x3_input_acceleration >= -3.198078319204689
 CBF_Constraint_for_table: x3_input_acceleration >= -4.034428204403947
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7598921807024144 x3_input_acceleration + relaxation_CLF
   <= -20.23097445051707
 CBF_Constraint_obs2: - 0.1236836101703812 x1_input_acceleration
   - 0.3114984117464084 x2_input_acceleration
   + 1.520911078766923 x3_input_acceleration >= -2.981874256826599
 CBF_Constraint_obs3: 0.2563163898296188 x1_input_acceleration
   - 0.3114984117464084 x2_input_acceleration
   + 1.520911078766923 x3_input_acceleration >= -3.044874827164738
 CBF_Constraint_obs1: - 0.4436836101703812 x1_input_acceleration
   - 0.3114984117464084 x2_input_acceleration
   + 1.520911078766923 x3_input_acceleration >= -3.208821144962904
 CBF_Constraint_for_table: x3_input_acceleration >= -4.041071701834452
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7612145024915243 x3_input_acceleration + relaxation_CLF
   <= -20.23177900752137
 CBF_Constraint_obs2: - 0.1237641956501175 x1_input_acceleration
   - 0.3115584932283419 x2_input_acceleration
   + 1.523502793420582 x3_input_acceleration >= -2.991808790484942
 CBF_Constraint_obs3: 0.2562257907695602 x1_input_acceleration
   - 0.3115277372316378 x2_input_acceleration
   + 1.523518409837514 x3_input_acceleration >= -3.054761915042471
 CBF_Constraint_obs1: - 0.4437742092304398 x1_input_acceleration
   - 0.3115277372316378 x2_input_acceleration
   + 1.523518409837514 x3_input_acceleration >= -3.21886678119574
 CBF_Constraint_for_table: x3_input_acceleration >= -4.047706854614688
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7625368242806343 x3_input_acceleration + relaxation_CLF
   <= -20.2325849633536
 CBF_Constraint_obs2: - 0.1237641956501175 x1_input_acceleration
   - 0.3115584932283419 x2_input_acceleration
   + 1.526147436998802 x3_input_acceleration >= -3.001890337857061
 CBF_Constraint_obs3: 0.2562358043498825 x1_input_acceleration
   - 0.3115584932283419 x2_input_acceleration
   + 1.526147436998802 x3_input_acceleration >= -3.064814351989449
 CBF_Constraint_obs1: - 0.4437641956501175 x1_input_acceleration
   - 0.3115584932283419 x2_input_acceleration
   + 1.526147436998802 x3_input_acceleration >= -3.228901694377155
 CBF_Constraint_for_table: x3_input_acceleration >= -4.054318463560238
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7638591460697443 x3_input_acceleration + relaxation_CLF
   <= -20.23339231801376
 CBF_Constraint_obs2: - 0.1237808849506546 x1_input_acceleration
   - 0.3115111671403903 x2_input_acceleration
   + 1.528817352946407 x3_input_acceleration >= -3.012054266752993
 CBF_Constraint_obs3: 0.2562191150493454 x1_input_acceleration
   - 0.3115111671403903 x2_input_acceleration
   + 1.528817352946407 x3_input_acceleration >= -3.074962426049871
 CBF_Constraint_obs1: - 0.4437808849506547 x1_input_acceleration
   - 0.3115111671403903 x2_input_acceleration
   + 1.528817352946407 x3_input_acceleration >= -3.239078974713517
 CBF_Constraint_for_table: x3_input_acceleration >= -4.060982822616414
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7651814678588542 x3_input_acceleration + relaxation_CLF
   <= -20.23420107150185
 CBF_Constraint_obs2: - 0.1238738681965043 x1_input_acceleration
   - 0.3118902526811618 x2_input_acceleration
   + 1.531153602092559 x3_input_acceleration >= -3.021314522675628
 CBF_Constraint_obs3: 0.2561261318034957 x1_input_acceleration
   - 0.3118902526811618 x2_input_acceleration
   + 1.531153602092559 x3_input_acceleration >= -3.084134347888949
 CBF_Constraint_obs1: - 0.4438738681965043 x1_input_acceleration
   - 0.3118902526811618 x2_input_acceleration
   + 1.531153602092559 x3_input_acceleration >= -3.248413617232831
 CBF_Constraint_for_table: x3_input_acceleration >= -4.067594431561964
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7665037896479642 x3_input_acceleration + relaxation_CLF
   <= -20.23501122381788
 CBF_Constraint_obs2: - 0.1237856533222367 x1_input_acceleration
   - 0.3115291677431125 x2_input_acceleration
   + 1.534117488148196 x3_input_acceleration >= -3.032362222206574
 CBF_Constraint_obs3: 0.2562143466777633 x1_input_acceleration
   - 0.3115291677431125 x2_input_acceleration
   + 1.534117488148196 x3_input_acceleration >= -3.095265851550449
 CBF_Constraint_obs1: - 0.4437856533222367 x1_input_acceleration
   - 0.3115291677431125 x2_input_acceleration
   + 1.534117488148196 x3_input_acceleration >= -3.259390744864364
 CBF_Constraint_for_table: x3_input_acceleration >= -4.074146137839515
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7678261114370741 x3_input_acceleration + relaxation_CLF
   <= -20.23582277496183
 CBF_Constraint_obs2: - 0.1238428737812211 x1_input_acceleration
   - 0.3118842922166842 x2_input_acceleration
   + 1.536472333943518 x3_input_acceleration >= -3.041695127614669
 CBF_Constraint_obs3: 0.256157126218779 x1_input_acceleration
   - 0.3118842922166842 x2_input_acceleration
   + 1.536472333943518 x3_input_acceleration >= -3.104544397522508
 CBF_Constraint_obs1: - 0.4438428737812211 x1_input_acceleration
   - 0.3118842922166842 x2_input_acceleration
   + 1.536472333943518 x3_input_acceleration >= -3.268769426639646
 CBF_Constraint_for_table: x3_input_acceleration >= -4.080746123879333
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7691484332261841 x3_input_acceleration + relaxation_CLF
   <= -20.23663572493372
 CBF_Constraint_obs2: - 0.123871007173555 x1_input_acceleration
   - 0.3118738017992038 x2_input_acceleration
   + 1.539109705755075 x3_input_acceleration >= -3.051834976270443
 CBF_Constraint_obs3: 0.256128992826445 x1_input_acceleration
   - 0.3118738017992038 x2_input_acceleration
   + 1.539109705755075 x3_input_acceleration >= -3.114657519455566
 CBF_Constraint_obs1: - 0.443871007173555 x1_input_acceleration
   - 0.3118738017992038 x2_input_acceleration
   + 1.539109705755075 x3_input_acceleration >= -3.278931782009287
 CBF_Constraint_for_table: x3_input_acceleration >= -4.087811324171623
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7704707550152941 x3_input_acceleration + relaxation_CLF
   <= -20.23745007373353
 CBF_Constraint_obs2: - 0.1237808849506546 x1_input_acceleration
   - 0.3115329824403781 x2_input_acceleration
   + 1.542038663488874 x3_input_acceleration >= -3.062822182902396
 CBF_Constraint_obs3: 0.2562191150493454 x1_input_acceleration
   - 0.3115329824403781 x2_input_acceleration
   + 1.542038663488874 x3_input_acceleration >= -3.125730342199274
 CBF_Constraint_obs1: - 0.4437808849506547 x1_input_acceleration
   - 0.3115329824403781 x2_input_acceleration
   + 1.542038663488874 x3_input_acceleration >= -3.289846890862919
 CBF_Constraint_for_table: x3_input_acceleration >= -4.093944307819627
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.771793076804404 x3_input_acceleration + relaxation_CLF
   <= -20.23826582136128
 CBF_Constraint_obs2: - 0.1238137867145707 x1_input_acceleration
   - 0.3119453273729342 x2_input_acceleration
   + 1.544419139281449 x3_input_acceleration >= -3.072337773536576
 CBF_Constraint_obs3: 0.2562272212810348 x1_input_acceleration
   - 0.3118499599412936 x2_input_acceleration
   + 1.544424384490189 x3_input_acceleration >= -3.135186832310859
 CBF_Constraint_obs1: - 0.4437727787189652 x1_input_acceleration
   - 0.3118499599412936 x2_input_acceleration
   + 1.544424384490189 x3_input_acceleration >= -3.299289195069048
 CBF_Constraint_for_table: x3_input_acceleration >= -4.100609858968698
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.773115398593514 x3_input_acceleration + relaxation_CLF
   <= -20.23908296781696
 CBF_Constraint_obs2: - 0.1238137867145707 x1_input_acceleration
   - 0.3119453273729342 x2_input_acceleration
   + 1.547063782859669 x3_input_acceleration >= -3.082557611608097
 CBF_Constraint_obs3: 0.2561862132854293 x1_input_acceleration
   - 0.3119453273729342 x2_input_acceleration
   + 1.547063782859669 x3_input_acceleration >= -3.145434514229255
 CBF_Constraint_obs1: - 0.4438137867145707 x1_input_acceleration
   - 0.3119453273729342 x2_input_acceleration
   + 1.547063782859669 x3_input_acceleration >= -3.309608640979754
 CBF_Constraint_for_table: x3_input_acceleration >= -4.107199116172458
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7744377203826239 x3_input_acceleration + relaxation_CLF
   <= -20.23990151310057
 CBF_Constraint_obs2: - 0.1238538410358597 x1_input_acceleration
   - 0.3119323335603732 x2_input_acceleration
   + 1.549685776672874 x3_input_acceleration >= -3.092709451352457
 CBF_Constraint_obs3: 0.2561461589641403 x1_input_acceleration
   - 0.3119323335603732 x2_input_acceleration
   + 1.549685776672874 x3_input_acceleration >= -3.15554830236839
 CBF_Constraint_obs1: - 0.4438538410358597 x1_input_acceleration
   - 0.3119323335603732 x2_input_acceleration
   + 1.549685776672874 x3_input_acceleration >= -3.319792524181145
 CBF_Constraint_for_table: x3_input_acceleration >= -4.114275641347255
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7757600421717339 x3_input_acceleration + relaxation_CLF
   <= -20.24072145721212
 CBF_Constraint_obs2: - 0.1236750271015336 x1_input_acceleration
   - 0.3115342937425631 x2_input_acceleration
   + 1.552706883187496 x3_input_acceleration >= -3.104059742005087
 CBF_Constraint_obs3: 0.2563249728984665 x1_input_acceleration
   - 0.3115342937425631 x2_input_acceleration
   + 1.552706883187496 x3_input_acceleration >= -3.167068466258631
 CBF_Constraint_obs1: - 0.4436750271015336 x1_input_acceleration
   - 0.3115342937425631 x2_input_acceleration
   + 1.552706883187496 x3_input_acceleration >= -3.330999763686314
 CBF_Constraint_for_table: x3_input_acceleration >= -4.120419651854542
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7770823639608438 x3_input_acceleration + relaxation_CLF
   <= -20.24154280015159
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.555248768358123 x3_input_acceleration >= -3.113969735900726
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.554976494340789 x3_input_acceleration >= -3.176029200041202
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.554976494340789 x3_input_acceleration >= -3.340313476175245
 CBF_Constraint_for_table: x3_input_acceleration >= -4.12749617702934
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7784046857499538 x3_input_acceleration + relaxation_CLF
   <= -20.24236554191899
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.557893411936343 x3_input_acceleration >= -3.124261175244728
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.557621137919009 x3_input_acceleration >= -3.186318839215875
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.557621137919009 x3_input_acceleration >= -3.350603115349919
 CBF_Constraint_for_table: x3_input_acceleration >= -4.13410778597489
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7797270075390638 x3_input_acceleration + relaxation_CLF
   <= -20.24318968251433
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.560538055514563 x3_input_acceleration >= -3.134570099937869
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.560265781497229 x3_input_acceleration >= -3.196625963739687
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.560265781497229 x3_input_acceleration >= -3.360910239873731
 CBF_Constraint_for_table: x3_input_acceleration >= -4.140719394920439
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7810493293281737 x3_input_acceleration + relaxation_CLF
   <= -20.24401522193759
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.563182699092783 x3_input_acceleration >= -3.14489650998015
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.562910425075449 x3_input_acceleration >= -3.206950573612639
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.562910425075449 x3_input_acceleration >= -3.371234849746683
 CBF_Constraint_for_table: x3_input_acceleration >= -4.14733100386599
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7823716511172837 x3_input_acceleration + relaxation_CLF
   <= -20.2448421601888
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.565827342671003 x3_input_acceleration >= -3.15524040537157
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.565555068653669 x3_input_acceleration >= -3.217292668834731
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.565555068653669 x3_input_acceleration >= -3.381576944968775
 CBF_Constraint_for_table: x3_input_acceleration >= -4.153942612811539
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7836939729063936 x3_input_acceleration + relaxation_CLF
   <= -20.24567049726792
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.568471986249222 x3_input_acceleration >= -3.16560178611213
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.568199712231888 x3_input_acceleration >= -3.227652249405963
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.568199712231888 x3_input_acceleration >= -3.391936525540006
 CBF_Constraint_for_table: x3_input_acceleration >= -4.160554221757089
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7850162946955036 x3_input_acceleration + relaxation_CLF
   <= -20.24650023317498
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.571116629827442 x3_input_acceleration >= -3.175980652201829
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.570844355810108 x3_input_acceleration >= -3.238029315326333
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.570844355810108 x3_input_acceleration >= -3.402313591460376
 CBF_Constraint_for_table: x3_input_acceleration >= -4.167165830702639
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7863386164846136 x3_input_acceleration + relaxation_CLF
   <= -20.24733136790998
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.573761273405662 x3_input_acceleration >= -3.186377003640668
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.573488999388328 x3_input_acceleration >= -3.248423866595843
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.573488999388328 x3_input_acceleration >= -3.412708142729887
 CBF_Constraint_for_table: x3_input_acceleration >= -4.173777439648188
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7876609382737235 x3_input_acceleration + relaxation_CLF
   <= -20.2481639014729
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.576405916983882 x3_input_acceleration >= -3.196790840428647
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.576133642966548 x3_input_acceleration >= -3.258835903214493
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.576133642966548 x3_input_acceleration >= -3.423120179348537
 CBF_Constraint_for_table: x3_input_acceleration >= -4.180389048593739
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7889832600628335 x3_input_acceleration + relaxation_CLF
   <= -20.24899783386375
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.579050560562102 x3_input_acceleration >= -3.207222162565765
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.578778286544768 x3_input_acceleration >= -3.269265425182282
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.578778286544768 x3_input_acceleration >= -3.433549701316326
 CBF_Constraint_for_table: x3_input_acceleration >= -4.187000657539288
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7903055818519434 x3_input_acceleration + relaxation_CLF
   <= -20.24983316508253
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.581695204140322 x3_input_acceleration >= -3.217670970052022
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.581422930122988 x3_input_acceleration >= -3.279712432499212
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.581422930122988 x3_input_acceleration >= -3.443996708633255
 CBF_Constraint_for_table: x3_input_acceleration >= -4.193612266484838
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7916279036410534 x3_input_acceleration + relaxation_CLF
   <= -20.25066989512925
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.584339847718542 x3_input_acceleration >= -3.22813726288742
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.584067573701208 x3_input_acceleration >= -3.29017692516528
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.584067573701208 x3_input_acceleration >= -3.454461201299324
 CBF_Constraint_for_table: x3_input_acceleration >= -4.200223875430388
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7929502254301634 x3_input_acceleration + relaxation_CLF
   <= -20.2515080240039
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.586984491296762 x3_input_acceleration >= -3.238621041071956
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.586712217279428 x3_input_acceleration >= -3.300658903180488
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.586712217279428 x3_input_acceleration >= -3.464943179314531
 CBF_Constraint_for_table: x3_input_acceleration >= -4.206835484375937
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7942725472192733 x3_input_acceleration + relaxation_CLF
   <= -20.25234755170648
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.589629134874982 x3_input_acceleration >= -3.249122304605632
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.589356860857648 x3_input_acceleration >= -3.311158366544836
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.589356860857648 x3_input_acceleration >= -3.47544264267888
 CBF_Constraint_for_table: x3_input_acceleration >= -4.213447093321488
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7955948690083833 x3_input_acceleration + relaxation_CLF
   <= -20.25318847823699
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.592273778453202 x3_input_acceleration >= -3.259641053488449
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.592001504435868 x3_input_acceleration >= -3.321675315258323
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.592001504435868 x3_input_acceleration >= -3.485959591392367
 CBF_Constraint_for_table: x3_input_acceleration >= -4.220058702267037
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7969171907974932 x3_input_acceleration + relaxation_CLF
   <= -20.25403080359543
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.594918422031422 x3_input_acceleration >= -3.270177287720404
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.594646148014088 x3_input_acceleration >= -3.332209749320949
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.594646148014088 x3_input_acceleration >= -3.496494025454994
 CBF_Constraint_for_table: x3_input_acceleration >= -4.226670311212587
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7982395125866032 x3_input_acceleration + relaxation_CLF
   <= -20.2548745277818
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.597563065609642 x3_input_acceleration >= -3.280731007301498
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.597290791592308 x3_input_acceleration >= -3.342761668732716
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.597290791592308 x3_input_acceleration >= -3.507045944866759
 CBF_Constraint_for_table: x3_input_acceleration >= -4.233281920158137
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.7995618343757132 x3_input_acceleration + relaxation_CLF
   <= -20.2557196507961
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.600207709187861 x3_input_acceleration >= -3.291302212231733
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.599935435170527 x3_input_acceleration >= -3.353331073493622
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.599935435170527 x3_input_acceleration >= -3.517615349627665
 CBF_Constraint_for_table: x3_input_acceleration >= -4.239893529103687
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8008841561648231 x3_input_acceleration + relaxation_CLF
   <= -20.25656617263834
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.602852352766081 x3_input_acceleration >= -3.301890902511107
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.602580078748747 x3_input_acceleration >= -3.363917963603667
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.602580078748747 x3_input_acceleration >= -3.528202239737711
 CBF_Constraint_for_table: x3_input_acceleration >= -4.246505138049236
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8022064779539331 x3_input_acceleration + relaxation_CLF
   <= -20.2574140933085
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.605496996344301 x3_input_acceleration >= -3.312497078139621
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.605224722326967 x3_input_acceleration >= -3.374522339062851
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.605224722326967 x3_input_acceleration >= -3.538806615196895
 CBF_Constraint_for_table: x3_input_acceleration >= -4.253116746994786
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.803528799743043 x3_input_acceleration + relaxation_CLF
   <= -20.2582634128066
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.608141639922521 x3_input_acceleration >= -3.323120739117273
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.607869365905187 x3_input_acceleration >= -3.385144199871176
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.607869365905187 x3_input_acceleration >= -3.54942847600522
 CBF_Constraint_for_table: x3_input_acceleration >= -4.259728355940336
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.804851121532153 x3_input_acceleration + relaxation_CLF
   <= -20.25911413113263
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.610786283500741 x3_input_acceleration >= -3.333761885444066
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.610514009483407 x3_input_acceleration >= -3.39578354602864
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.610514009483407 x3_input_acceleration >= -3.560067822162684
 CBF_Constraint_for_table: x3_input_acceleration >= -4.266339964885885
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.806173443321263 x3_input_acceleration + relaxation_CLF
   <= -20.25996624828658
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.613430927078961 x3_input_acceleration >= -3.344420517119998
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.613158653061627 x3_input_acceleration >= -3.406440377535244
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.613158653061627 x3_input_acceleration >= -3.570724653669287
 CBF_Constraint_for_table: x3_input_acceleration >= -4.272951573831436
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8074957651103729 x3_input_acceleration + relaxation_CLF
   <= -20.26081976426848
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.616075570657181 x3_input_acceleration >= -3.35509663414507
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.615803296639847 x3_input_acceleration >= -3.417114694390987
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.615803296639847 x3_input_acceleration >= -3.58139897052503
 CBF_Constraint_for_table: x3_input_acceleration >= -4.279563182776985
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8088180868994829 x3_input_acceleration + relaxation_CLF
   <= -20.2616746790783
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.618720214235401 x3_input_acceleration >= -3.365790236519282
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.618447940218067 x3_input_acceleration >= -3.427806496595869
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.618447940218067 x3_input_acceleration >= -3.592090772729914
 CBF_Constraint_for_table: x3_input_acceleration >= -4.286174791722535
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8101404086885928 x3_input_acceleration + relaxation_CLF
   <= -20.26253099271605
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.621364857813621 x3_input_acceleration >= -3.376501324242632
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.621092583796287 x3_input_acceleration >= -3.438515784149891
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.621092583796287 x3_input_acceleration >= -3.602800060283935
 CBF_Constraint_for_table: x3_input_acceleration >= -4.292786400668085
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8114627304777028 x3_input_acceleration + relaxation_CLF
   <= -20.26338870518173
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.624009501391841 x3_input_acceleration >= -3.387229897315123
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.623737227374507 x3_input_acceleration >= -3.449242557053053
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.623737227374507 x3_input_acceleration >= -3.613526833187097
 CBF_Constraint_for_table: x3_input_acceleration >= -4.299398009613634
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8127850522668127 x3_input_acceleration + relaxation_CLF
   <= -20.26424781647535
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.626654144970061 x3_input_acceleration >= -3.397975955736752
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.626381870952727 x3_input_acceleration >= -3.459986815305355
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.626381870952727 x3_input_acceleration >= -3.624271091439399
 CBF_Constraint_for_table: x3_input_acceleration >= -4.306009618559185
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8141073740559227 x3_input_acceleration + relaxation_CLF
   <= -20.26510832659689
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.629298788548281 x3_input_acceleration >= -3.408739499507522
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.629026514530947 x3_input_acceleration >= -3.470748558906795
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.629026514530947 x3_input_acceleration >= -3.635032835040839
 CBF_Constraint_for_table: x3_input_acceleration >= -4.312621227504734
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8154296958450327 x3_input_acceleration + relaxation_CLF
   <= -20.26597023554637
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.631943432126501 x3_input_acceleration >= -3.419520528627431
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.631671158109167 x3_input_acceleration >= -3.481527787857376
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.631671158109167 x3_input_acceleration >= -3.64581206399142
 CBF_Constraint_for_table: x3_input_acceleration >= -4.319232836450285
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8167520176341426 x3_input_acceleration + relaxation_CLF
   <= -20.26683354332378
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.63458807570472 x3_input_acceleration >= -3.43031904309648
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.634315801687386 x3_input_acceleration >= -3.492324502157096
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.634315801687386 x3_input_acceleration >= -3.656608778291139
 CBF_Constraint_for_table: x3_input_acceleration >= -4.325844445395834
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8180743394232526 x3_input_acceleration + relaxation_CLF
   <= -20.26769824992911
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.63723271928294 x3_input_acceleration >= -3.441135042914667
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.636960445265606 x3_input_acceleration >= -3.503138701805955
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.636960445265606 x3_input_acceleration >= -3.667422977939999
 CBF_Constraint_for_table: x3_input_acceleration >= -4.332456054341383
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8193966612123625 x3_input_acceleration + relaxation_CLF
   <= -20.26856435536239
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.63987736286116 x3_input_acceleration >= -3.451968528081996
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.639605088843826 x3_input_acceleration >= -3.513970386803954
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.639605088843826 x3_input_acceleration >= -3.678254662937998
 CBF_Constraint_for_table: x3_input_acceleration >= -4.339067663286934
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8207189830014725 x3_input_acceleration + relaxation_CLF
   <= -20.26943185962359
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.64252200643938 x3_input_acceleration >= -3.462819498598463
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.642249732422046 x3_input_acceleration >= -3.524819557151093
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.642249732422046 x3_input_acceleration >= -3.689103833285137
 CBF_Constraint_for_table: x3_input_acceleration >= -4.345679272232483
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8220413047905825 x3_input_acceleration + relaxation_CLF
   <= -20.27030076271272
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.6451666500176 x3_input_acceleration >= -3.473687954464069
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.644894376000266 x3_input_acceleration >= -3.535686212847371
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.644894376000266 x3_input_acceleration >= -3.699970488981414
 CBF_Constraint_for_table: x3_input_acceleration >= -4.352290881178033
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8233636265796924 x3_input_acceleration + relaxation_CLF
   <= -20.27117106462979
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.64781129359582 x3_input_acceleration >= -3.484573895678816
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.647539019578486 x3_input_acceleration >= -3.546570353892788
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.647539019578486 x3_input_acceleration >= -3.710854630026832
 CBF_Constraint_for_table: x3_input_acceleration >= -4.358902490123583
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8246859483688024 x3_input_acceleration + relaxation_CLF
   <= -20.27204276537478
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.65045593717404 x3_input_acceleration >= -3.495477322242702
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.650183663156706 x3_input_acceleration >= -3.557471980287346
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.650183663156706 x3_input_acceleration >= -3.72175625642139
 CBF_Constraint_for_table: x3_input_acceleration >= -4.365514099069133
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8260082701579123 x3_input_acceleration + relaxation_CLF
   <= -20.27291586494771
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.65310058075226 x3_input_acceleration >= -3.506398234155727
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.652828306734926 x3_input_acceleration >= -3.568391092031042
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.652828306734926 x3_input_acceleration >= -3.732675368165086
 CBF_Constraint_for_table: x3_input_acceleration >= -4.372125708014682
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8273305919470223 x3_input_acceleration + relaxation_CLF
   <= -20.27379036334856
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.65574522433048 x3_input_acceleration >= -3.517336631417892
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.655472950313146 x3_input_acceleration >= -3.579327689123879
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.655472950313146 x3_input_acceleration >= -3.743611965257923
 CBF_Constraint_for_table: x3_input_acceleration >= -4.378737316960232
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8286529137361323 x3_input_acceleration + relaxation_CLF
   <= -20.27466626057735
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.6583898679087 x3_input_acceleration >= -3.528292514029197
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.658117593891366 x3_input_acceleration >= -3.590281771565855
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.658117593891366 x3_input_acceleration >= -3.754566047699899
 CBF_Constraint_for_table: x3_input_acceleration >= -4.385348925905782
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8299752355252422 x3_input_acceleration + relaxation_CLF
   <= -20.27554355663407
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.66103451148692 x3_input_acceleration >= -3.539265881989641
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.660762237469586 x3_input_acceleration >= -3.60125333935697
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.660762237469586 x3_input_acceleration >= -3.765537615491014
 CBF_Constraint_for_table: x3_input_acceleration >= -4.391960534851332
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8312975573143522 x3_input_acceleration + relaxation_CLF
   <= -20.27642225151872
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.66367915506514 x3_input_acceleration >= -3.550256735299224
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.663406881047806 x3_input_acceleration >= -3.612242392497225
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.663406881047806 x3_input_acceleration >= -3.776526668631269
 CBF_Constraint_for_table: x3_input_acceleration >= -4.398572143796882
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8326198791034621 x3_input_acceleration + relaxation_CLF
   <= -20.27730234523131
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.666323798643359 x3_input_acceleration >= -3.561265073957948
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.666051524626025 x3_input_acceleration >= -3.62324893098662
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.666051524626025 x3_input_acceleration >= -3.787533207120664
 CBF_Constraint_for_table: x3_input_acceleration >= -4.405183752742431
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8339422008925721 x3_input_acceleration + relaxation_CLF
   <= -20.27818383777182
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.668968442221579 x3_input_acceleration >= -3.57229089796581
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.668696168204245 x3_input_acceleration >= -3.634272954825154
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.668696168204245 x3_input_acceleration >= -3.798557230959197
 CBF_Constraint_for_table: x3_input_acceleration >= -4.411795361687981
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8352645226816821 x3_input_acceleration + relaxation_CLF
   <= -20.27906672914026
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.671613085799799 x3_input_acceleration >= -3.583334207322812
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.671340811782465 x3_input_acceleration >= -3.645314464012828
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.671340811782465 x3_input_acceleration >= -3.809598740146871
 CBF_Constraint_for_table: x3_input_acceleration >= -4.418406970633531
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.836586844470792 x3_input_acceleration + relaxation_CLF
   <= -20.27995101933664
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.674257729378019 x3_input_acceleration >= -3.594395002028954
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.673985455360685 x3_input_acceleration >= -3.656373458549641
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.673985455360685 x3_input_acceleration >= -3.820657734683684
 CBF_Constraint_for_table: x3_input_acceleration >= -4.425018579579081
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.837909166259902 x3_input_acceleration + relaxation_CLF
   <= -20.28083670836094
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.676902372956239 x3_input_acceleration >= -3.605473282084235
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.676630098938905 x3_input_acceleration >= -3.667449938435594
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.676630098938905 x3_input_acceleration >= -3.831734214569638
 CBF_Constraint_for_table: x3_input_acceleration >= -4.431630188524631
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8392314880490119 x3_input_acceleration + relaxation_CLF
   <= -20.28172379621318
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.679547016534459 x3_input_acceleration >= -3.616569047488656
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.679274742517125 x3_input_acceleration >= -3.678543903670686
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.679274742517125 x3_input_acceleration >= -3.842828179804729
 CBF_Constraint_for_table: x3_input_acceleration >= -4.43824179747018
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8405538098381219 x3_input_acceleration + relaxation_CLF
   <= -20.28261228289335
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.682191660112679 x3_input_acceleration >= -3.627682298242217
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.681919386095345 x3_input_acceleration >= -3.689655354254918
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.681919386095345 x3_input_acceleration >= -3.853939630388962
 CBF_Constraint_for_table: x3_input_acceleration >= -4.444853406415731
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8418761316272318 x3_input_acceleration + relaxation_CLF
   <= -20.28350216840145
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.684836303690899 x3_input_acceleration >= -3.638813034344917
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.684564029673565 x3_input_acceleration >= -3.700784290188289
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.684564029673565 x3_input_acceleration >= -3.865068566322333
 CBF_Constraint_for_table: x3_input_acceleration >= -4.45146501536128
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8431984534163418 x3_input_acceleration + relaxation_CLF
   <= -20.28439345273748
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.687480947269119 x3_input_acceleration >= -3.649961255796756
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.687208673251785 x3_input_acceleration >= -3.7119307114708
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.687208673251785 x3_input_acceleration >= -3.876214987604843
 CBF_Constraint_for_table: x3_input_acceleration >= -4.458076624306829
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8445207752054518 x3_input_acceleration + relaxation_CLF
   <= -20.28528613590145
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.690125590847339 x3_input_acceleration >= -3.661126962597736
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.689853316830005 x3_input_acceleration >= -3.723094618102451
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.689853316830005 x3_input_acceleration >= -3.887378894236494
 CBF_Constraint_for_table: x3_input_acceleration >= -4.46468823325238
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8458430969945617 x3_input_acceleration + relaxation_CLF
   <= -20.28618021789334
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.692770234425559 x3_input_acceleration >= -3.672310154747854
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.692497960408225 x3_input_acceleration >= -3.73427601008324
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.692497960408225 x3_input_acceleration >= -3.898560286217284
 CBF_Constraint_for_table: x3_input_acceleration >= -4.471299842197929
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8471654187836717 x3_input_acceleration + relaxation_CLF
   <= -20.28707569871317
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.695414878003779 x3_input_acceleration >= -3.683510832247113
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.695142603986445 x3_input_acceleration >= -3.74547488741317
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.695142603986445 x3_input_acceleration >= -3.909759163547214
 CBF_Constraint_for_table: x3_input_acceleration >= -4.477911451143479
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8484877405727816 x3_input_acceleration + relaxation_CLF
   <= -20.28797257836092
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.698059521581998 x3_input_acceleration >= -3.69472899509551
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.697787247564664 x3_input_acceleration >= -3.756691250092239
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.697787247564664 x3_input_acceleration >= -3.920975526226282
 CBF_Constraint_for_table: x3_input_acceleration >= -4.484523060089029
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8498100623618916 x3_input_acceleration + relaxation_CLF
   <= -20.28887085683661
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.700704165160218 x3_input_acceleration >= -3.705964643293048
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.700431891142884 x3_input_acceleration >= -3.767925098120448
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.700431891142884 x3_input_acceleration >= -3.932209374254492
 CBF_Constraint_for_table: x3_input_acceleration >= -4.491134669034579
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8511323841510016 x3_input_acceleration + relaxation_CLF
   <= -20.28977053414023
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.703348808738438 x3_input_acceleration >= -3.717217776839725
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.703076534721104 x3_input_acceleration >= -3.779176431497796
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.703076534721104 x3_input_acceleration >= -3.943460707631839
 CBF_Constraint_for_table: x3_input_acceleration >= -4.497746277980129
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8524547059401115 x3_input_acceleration + relaxation_CLF
   <= -20.29067161027178
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.705993452316658 x3_input_acceleration >= -3.728488395735541
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.705721178299324 x3_input_acceleration >= -3.790445250224284
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.705721178299324 x3_input_acceleration >= -3.954729526358327
 CBF_Constraint_for_table: x3_input_acceleration >= -4.504357886925678
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8537770277292215 x3_input_acceleration + relaxation_CLF
   <= -20.29157408523126
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.708638095894878 x3_input_acceleration >= -3.739776499980497
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.708365821877544 x3_input_acceleration >= -3.801731554299911
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.708365821877544 x3_input_acceleration >= -3.966015830433955
 CBF_Constraint_for_table: x3_input_acceleration >= -4.510969495871228
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8550993495183314 x3_input_acceleration + relaxation_CLF
   <= -20.29247795901867
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.711282739473098 x3_input_acceleration >= -3.751082089574593
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.711010465455764 x3_input_acceleration >= -3.813035343724678
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.711010465455764 x3_input_acceleration >= -3.977319619858721
 CBF_Constraint_for_table: x3_input_acceleration >= -4.517581104816778
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8564216713074414 x3_input_acceleration + relaxation_CLF
   <= -20.29338323163401
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.713927383051318 x3_input_acceleration >= -3.762405164517828
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.713655109033984 x3_input_acceleration >= -3.824356618498585
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.713655109033984 x3_input_acceleration >= -3.988640894632628
 CBF_Constraint_for_table: x3_input_acceleration >= -4.524192713762328
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8577439930965514 x3_input_acceleration + relaxation_CLF
   <= -20.29428990307729
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.716572026629538 x3_input_acceleration >= -3.773745724810202
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.716299752612204 x3_input_acceleration >= -3.83569537862163
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.716299752612204 x3_input_acceleration >= -3.999979654755674
 CBF_Constraint_for_table: x3_input_acceleration >= -4.530804322707878
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8590663148856613 x3_input_acceleration + relaxation_CLF
   <= -20.29519797334849
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.719216670207758 x3_input_acceleration >= -3.785103770451717
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.718944396190424 x3_input_acceleration >= -3.847051624093816
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.718944396190424 x3_input_acceleration >= -4.011335900227859
 CBF_Constraint_for_table: x3_input_acceleration >= -4.537415931653427
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8603886366747713 x3_input_acceleration + relaxation_CLF
   <= -20.29610744244763
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.721861313785978 x3_input_acceleration >= -3.79647930144237
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.721589039768644 x3_input_acceleration >= -3.858425354915141
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.721589039768644 x3_input_acceleration >= -4.022709631049185
 CBF_Constraint_for_table: x3_input_acceleration >= -4.544027540598977
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8617109584638812 x3_input_acceleration + relaxation_CLF
   <= -20.2970183103747
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.724505957364198 x3_input_acceleration >= -3.807872317782163
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.724233683346864 x3_input_acceleration >= -3.869816571085606
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.724233683346864 x3_input_acceleration >= -4.03410084721965
 CBF_Constraint_for_table: x3_input_acceleration >= -4.550639149544527
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8630332802529912 x3_input_acceleration + relaxation_CLF
   <= -20.29793057712969
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.727150600942418 x3_input_acceleration >= -3.819282819471097
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.726878326925084 x3_input_acceleration >= -3.88122527260521
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.726878326925084 x3_input_acceleration >= -4.045509548739253
 CBF_Constraint_for_table: x3_input_acceleration >= -4.557250758490077
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8643556020421012 x3_input_acceleration + relaxation_CLF
   <= -20.29884424271263
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.729795244520637 x3_input_acceleration >= -3.830710806509169
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.729522970503304 x3_input_acceleration >= -3.892651459473954
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.729522970503304 x3_input_acceleration >= -4.056935735607997
 CBF_Constraint_for_table: x3_input_acceleration >= -4.563862367435626
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8656779238312111 x3_input_acceleration + relaxation_CLF
   <= -20.29975930712349
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.732439888098857 x3_input_acceleration >= -3.84215627889638
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.732167614081523 x3_input_acceleration >= -3.904095131691836
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.732167614081523 x3_input_acceleration >= -4.068379407825881
 CBF_Constraint_for_table: x3_input_acceleration >= -4.570473976381177
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8670002456203211 x3_input_acceleration + relaxation_CLF
   <= -20.30067577036228
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.735084531677077 x3_input_acceleration >= -3.853619236632732
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.734812257659743 x3_input_acceleration >= -3.91555628925886
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.734812257659743 x3_input_acceleration >= -4.079840565392904
 CBF_Constraint_for_table: x3_input_acceleration >= -4.577085585326726
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.868322567409431 x3_input_acceleration + relaxation_CLF
   <= -20.301593632429
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.737729175255297 x3_input_acceleration >= -3.865099679718223
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.737456901237963 x3_input_acceleration >= -3.927034932175022
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.737456901237963 x3_input_acceleration >= -4.091319208309065
 CBF_Constraint_for_table: x3_input_acceleration >= -4.583697194272276
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.869644889198541 x3_input_acceleration + relaxation_CLF
   <= -20.30251289332366
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.740373818833517 x3_input_acceleration >= -3.876597608152854
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.740101544816183 x3_input_acceleration >= -3.938531060440324
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.740101544816183 x3_input_acceleration >= -4.102815336574368
 CBF_Constraint_for_table: x3_input_acceleration >= -4.590308803217826
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.870967210987651 x3_input_acceleration + relaxation_CLF
   <= -20.30343355304624
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.743018462411737 x3_input_acceleration >= -3.888113021936624
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.742746188394403 x3_input_acceleration >= -3.950044674054765
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.742746188394403 x3_input_acceleration >= -4.114328950188809
 CBF_Constraint_for_table: x3_input_acceleration >= -4.596920412163375
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8722895327767609 x3_input_acceleration + relaxation_CLF
   <= -20.30435561159676
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.745663105989957 x3_input_acceleration >= -3.899645921069533
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.745390831972623 x3_input_acceleration >= -3.961575773018347
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.745390831972623 x3_input_acceleration >= -4.12586004915239
 CBF_Constraint_for_table: x3_input_acceleration >= -4.603532021108926
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8736118545658709 x3_input_acceleration + relaxation_CLF
   <= -20.30527906897521
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.748307749568177 x3_input_acceleration >= -3.911196305551583
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.748035475550843 x3_input_acceleration >= -3.973124357331067
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.748035475550843 x3_input_acceleration >= -4.13740863346511
 CBF_Constraint_for_table: x3_input_acceleration >= -4.610143630054475
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8749341763549808 x3_input_acceleration + relaxation_CLF
   <= -20.30620392518159
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.750952393146397 x3_input_acceleration >= -3.922764175382771
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.750680119129063 x3_input_acceleration >= -3.984690426992927
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.750680119129063 x3_input_acceleration >= -4.148974703126971
 CBF_Constraint_for_table: x3_input_acceleration >= -4.616755239000025
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8762564981440908 x3_input_acceleration + relaxation_CLF
   <= -20.3071301802159
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.753597036724617 x3_input_acceleration >= -3.9343495305631
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.753324762707283 x3_input_acceleration >= -3.996273982003927
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.753324762707283 x3_input_acceleration >= -4.160558258137971
 CBF_Constraint_for_table: x3_input_acceleration >= -4.623366847945575
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8775788199332007 x3_input_acceleration + relaxation_CLF
   <= -20.30805783407814
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.756241680302837 x3_input_acceleration >= -3.945952371092567
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.755969406285503 x3_input_acceleration >= -4.007875022364066
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.755969406285503 x3_input_acceleration >= -4.17215929849811
 CBF_Constraint_for_table: x3_input_acceleration >= -4.629978456891124
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8789011417223107 x3_input_acceleration + relaxation_CLF
   <= -20.30898688676831
 CBF_Constraint_obs2: - 0.123774686067598 x1_input_acceleration
   - 0.3115395389513034 x2_input_acceleration
   + 1.758886323881057 x3_input_acceleration >= -3.957572696971175
 CBF_Constraint_obs3: 0.2561232707805465 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.758614049863723 x3_input_acceleration >= -4.019493548073345
 CBF_Constraint_obs1: - 0.4438767292194535 x1_input_acceleration
   - 0.3118851266817111 x2_input_acceleration
   + 1.758614049863723 x3_input_acceleration >= -4.183777824207389
 CBF_Constraint_for_table: x3_input_acceleration >= -4.636590065836675
 Relaxed_CLF_constraint: - 1.21012e-08 x1_input_acceleration
   - 1.4589e-08 x2_input_acceleration
   + 0.8802234635114207 x3_input_acceleration + relaxation_CLF
   <= -20.30991733828642
Bounds
 -0.1 <= x1_input_acceleration <= 0.1
 -0.1 <= x2_input_acceleration <= 0.1
 -0.1 <= x3_input_acceleration <= 0.1
 -30 <= relaxation_CLF <= 30
End
