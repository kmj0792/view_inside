#app.py
from flask import Flask, request, render_template, jsonify,redirect, json,url_for
from flaskext.mysql import MySQL #pip install flask-mysql
import pymysql
import torch
import torch.nn as nn
import pandas as pd
#import torchvision
#import torchvision.transforms as transforms
from dataset import datasetTest
  
app = Flask(__name__)
    
mysql = MySQL()
   
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = 'alswjd1004'
app.config['MYSQL_DATABASE_DB'] = 'viewinside'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
mysql.init_app(app)
  
folder_path = 'C:/Users/user/anaconda3/viewinside'
checkpoint_model = 'C:/Users/user/anaconda3/viewinside/model/model-8420.pth'

# Hyper-parameters 
input_size = 4
batch_size = 4096*4


# Test CSV dataset
test_dataset = datasetTest(folder_path)
# Data loader
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)
class NeuralNet(nn.Module):
    '''
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 2048, bias=True) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2048, 2, bias=True)  
    
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    '''
    def __init__(self,input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024, bias=True) 
        self.fc2 = nn.Linear(1024, 2048, bias=True) 
        self.fc3 = nn.Linear(2048, 4096, bias=True)
        self.fc4_1 = nn.Linear(4096, 2048, bias=True)
        self.fc4_2 = nn.Linear(4096, 2048, bias=True)
        self.fc5_1 = nn.Linear(2048, 1, bias=True)
        self.fc5_2 = nn.Linear(2048, 1, bias=True)
        self.relu = nn.ReLU()

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.xavier_uniform_(self.fc4_1.weight)
        torch.nn.init.xavier_uniform_(self.fc4_2.weight)
        torch.nn.init.xavier_uniform_(self.fc5_1.weight)
        torch.nn.init.xavier_uniform_(self.fc5_2.weight)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.relu(out)

        out = self.fc3(out)
        out = self.relu(out)

        out_1 = self.fc4_1(out)
        out_1 = self.relu(out_1)
        out_1 = self.fc5_1(out_1)

        out_2 = self.fc4_2(out)
        out_2 = self.relu(out_2)
        out_2 = self.fc5_2(out_2)

        out = torch.cat([out_1, out_2], dim = 1)

        return out

#model = NeuralNet(input_size)
model = NeuralNet(input_size)
model.load_state_dict(torch.load(checkpoint_model, map_location ='cpu')) #trained 
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/keyword/')
def keyword():
    return render_template('keyword.html')

@app.route('/locate/')
def locate():
    return render_template('locate.html')



@app.route('/market/', methods=['GET', 'POST'])
def market():
    
    if request.method == 'GET':
        return render_template('market.html')

    if request.method == 'POST':
        quarter_or= int(request.form['quarter'])
        area_or = (request.form['area'])
        detail_area_or = int(request.form['detail_area'])
        sector_or =(request.form['sector'])

        #분기
        quarter = (quarter_or-1)/(4-1)

        #상권
        if area_or=='A':
            area=1
            area_ko="골목상권"
        elif area_or=='D':
            area=2
            area_ko="발달상권"
        elif area_or=='R':
            area=3
            area_ko="전통시장"
        elif area_or=='U':
            area=4
            area_ko="관광특구"

        area=(area-1)/(4-1)
        
        #위치
        if detail_area_or == 1001060:
            detail_area_ko="가락시장"
        if detail_area_or == 1000507:
            detail_area_ko="가로공원로58길"
        if detail_area_or == 1000552:
            detail_area_ko="가로공원로76가길"
        if detail_area_or == 1000553:
            detail_area_ko="가로공원로80길"
        if detail_area_or == 1001419:
            detail_area_ko="가리봉시장"
        if detail_area_or == 1000683:
            detail_area_ko="가마산로61길"
        if detail_area_or == 1000566:
            detail_area_ko="강서로8길"
        if detail_area_or == 1001021:
            detail_area_ko="가산디지털단지역_1"
        if detail_area_or == 1001023:
            detail_area_ko="가산디지털단지역_2"
        if detail_area_or == 1001027:
            detail_area_ko="가산디지털단지역_3"
        if detail_area_or == 1000654:
            detail_area_ko="가산로3길"
        if detail_area_or == 1000655:
            detail_area_ko="가산로5길"
        if detail_area_or == 1000435:
            detail_area_ko="가재울로6길"
        if detail_area_or == 1000389:
            detail_area_ko="가좌로7길"
        if detail_area_or == 1000390:
            detail_area_ko="갈현로1길"
        if detail_area_or == 1000391:
            detail_area_ko="갈현로33길"
        if detail_area_or == 1000392:
            detail_area_ko="갈현로41길"
        if detail_area_or == 1000393:
            detail_area_ko="갈현로7길"
        if detail_area_or == 1001369:
            detail_area_ko="갈현시장"
        if detail_area_or == 1001496:
            detail_area_ko="강남 마이스 관광특구"
        if detail_area_or == 1001475:
            detail_area_ko="강남개포시장"
        if detail_area_or == 1001462:
            detail_area_ko="강남골목시장"
        if detail_area_or == 1001092:
            detail_area_ko="강남구 논현역_1"
        if detail_area_or == 1001095:
            detail_area_ko="강남구 논현역_2"
        if detail_area_or == 1001100:
            detail_area_ko="강남구 논현역_3"
        if detail_area_or == 1001109:
            detail_area_ko="강남구 논현역_4"
        if detail_area_or == 1001114:
            detail_area_ko="강남구 신사역_1"
        if detail_area_or == 1001117:
            detail_area_ko="강남구 신사역_2"
        if detail_area_or == 1000886:
            detail_area_ko="강남대로118길"
        if detail_area_or == 1000887:
            detail_area_ko="강남대로136길"
        if detail_area_or == 1000888:
            detail_area_ko="강남대로140길"
        if detail_area_or == 1000844:
            detail_area_ko="강남대로23길"
        if detail_area_or == 1000845:
            detail_area_ko="강남대로34길"
        if detail_area_or == 1000846:
            detail_area_ko="강남대로8길"
        if detail_area_or == 1001470:
            detail_area_ko="강남시장_강남"
        if detail_area_or == 1001443:
            detail_area_ko="강남시장_동작"
        if detail_area_or == 1000967:
            detail_area_ko="강동대로53길"
        if detail_area_or == 1001355:
            detail_area_ko="강북종합시장"
        if detail_area_or == 1001179:
            detail_area_ko="강서구청"
        if detail_area_or == 1000554:
            detail_area_ko="강서로15길"
        if detail_area_or == 1000555:
            detail_area_ko="강서로17가길"
        if detail_area_or == 1000556:
            detail_area_ko="강서로18길"
        if detail_area_or == 1000557:
            detail_area_ko="강서로29길"
        if detail_area_or == 1000558:
            detail_area_ko="강서로35길"
        if detail_area_or == 1000559:
            detail_area_ko="강서로45길"
        if detail_area_or == 1000560:
            detail_area_ko="강서로45다길"
        if detail_area_or == 1000561:
            detail_area_ko="강서로47길"
        if detail_area_or == 1000562:
            detail_area_ko="강서로47라길"
        if detail_area_or == 1000563:
            detail_area_ko="강서로5가길"
        if detail_area_or == 1000564:
            detail_area_ko="강서로5나길"
        if detail_area_or == 1000565:
            detail_area_ko="강서로5라길"
        if detail_area_or == 1000566:
            detail_area_ko="강서로8길"
        if detail_area_or == 1001049:
            detail_area_ko="개봉동 현대아파트 인근"
        if detail_area_or == 1000609:
            detail_area_ko="개봉로11길"
        if detail_area_or == 1000610:
            detail_area_ko="개봉로17길"
        if detail_area_or == 1000611:
            detail_area_ko="개봉로17다길"
        if detail_area_or == 1000612:
            detail_area_ko="개봉로1길"
        if detail_area_or == 1000613:
            detail_area_ko="개봉로20길"
        if detail_area_or == 1000614:
            detail_area_ko="개봉로23가길"
        if detail_area_or == 1000615:
            detail_area_ko="개봉로2길"
        if detail_area_or == 1000616:
            detail_area_ko="개봉로3길"
        if detail_area_or == 1001421:
            detail_area_ko="개봉중앙시장"
        if detail_area_or == 1000291:
            detail_area_ko="개운사길"
        if detail_area_or == 1000889:
            detail_area_ko="개포로17길"
        if detail_area_or == 1000890:
            detail_area_ko="개포로20길"
        if detail_area_or == 1000891:
            detail_area_ko="개포로28길"
        if detail_area_or == 1000892:
            detail_area_ko="개포로82길"
        if detail_area_or == 1000567:
            detail_area_ko="개화동로23길"
        if detail_area_or == 1000568:
            detail_area_ko="개화동로25길"
        if detail_area_or == 1000940:
            detail_area_ko="거마로20길"
        if detail_area_or == 1001161:
            detail_area_ko="건대입구역"
        if detail_area_or == 1000222:
            detail_area_ko="겸재로29길"
        if detail_area_or == 1000223:
            detail_area_ko="겸재로36길"
        if detail_area_or == 1000224:
            detail_area_ko="겸재로54길"
        if detail_area_or == 1001314:
            detail_area_ko="경동시장"
        if detail_area_or == 1000617:
            detail_area_ko="경인로15길"
        if detail_area_or == 1000618:
            detail_area_ko="경인로31길"
        if detail_area_or == 1000619:
            detail_area_ko="경인로35길"
        if detail_area_or == 1000620:
            detail_area_ko="경인로47길"
        if detail_area_or == 1000684:
            detail_area_ko="경인로77길"
        if detail_area_or == 1000685:
            detail_area_ko="경인로80길"
        if detail_area_or == 1000686:
            detail_area_ko="경인로90길"
        if detail_area_or == 1001399:
            detail_area_ko="경창시장"
        if detail_area_or == 1000168:
            detail_area_ko="경희대로3길"
        if detail_area_or == 1001273:
            detail_area_ko="남대문시장" #여기부터 추가함(11/1)
        if detail_area_or == 1000854:
            detail_area_ko="동광로1길"
        if detail_area_or == 1001493:
            detail_area_ko="동대문패션타운 관광특구"
        if detail_area_or == 1001492:
            detail_area_ko="명동 남대문 북창동 다동 무교동 관광특구"
        if detail_area_or == 1001491:
            detail_area_ko="이태원 관광특구"
        if detail_area_or == 1001495:
            detail_area_ko="잠실 관광특구"
        if detail_area_or == 1001494:
            detail_area_ko="종로·청계 관광특구"
        if detail_area_or == 1001253:
            detail_area_ko="서울 은평구 연신내역_1"
        if detail_area_or == 1001234:
            detail_area_ko="서울 종로구 경복궁역_1"
        if detail_area_or == 1001215:
            detail_area_ko="서울 종로구 광화문역_1"
        if detail_area_or == 1001219:
            detail_area_ko="서울 종로구 동대문역_1"
        if detail_area_or == 1001189:
            detail_area_ko="서울 중구 명동역"
        if detail_area_or == 1001192:
            detail_area_ko="서울 중구 충무로역_1"
        if detail_area_or == 1001180:
            detail_area_ko="신촌역"
        if detail_area_or == 1001138:
            detail_area_ko="압구정 로데오거리_1"
        if detail_area_or == 1001110:
            detail_area_ko="잠실역"
        if detail_area_or == 1001097:
            detail_area_ko="잠실역 롯데월드" 
        if detail_area_or == 1001450:
            detail_area_ko="남성역골목시장"
        if detail_area_or == 1001426:
            detail_area_ko="대명시장"
        if detail_area_or == 1001481:
            detail_area_ko="명일골목시장"
        if detail_area_or == 1001479:
            detail_area_ko="석촌시장"
        if detail_area_or == 1001488:
            detail_area_ko="성내골목시장"
        if detail_area_or == 1001386:
            detail_area_ko="신수시장"
        if detail_area_or == 1001402:
            detail_area_ko="신월6동골목시장"
        if detail_area_or == 1001483:
            detail_area_ko="암사종합시장"
        if detail_area_or == 1001484:
            detail_area_ko="양지골목시장"
        if detail_area_or == 1001097:
            detail_area_ko="역촌중앙시장"
        if detail_area_or == 1001471:
            detail_area_ko="영동전통시장"
        if detail_area_or == 1001452:
            detail_area_ko="중부시장"
        if detail_area_or == 1001412:
            detail_area_ko="화곡본동시장"
        if detail_area_or == 1001409:
            detail_area_ko="화곡중앙골목식당"
        if detail_area_or == 1001300:
            detail_area_ko="화장제일골목시장"
        if detail_area_or == 1001326:
            detail_area_ko="회기시장"
        if detail_area_or == 1001286:
            detail_area_ko="후암시장"
        if detail_area_or == 100148:
            detail_area_ko="흑석시장"
        if detail_area_or == 1000838:
            detail_area_ko="호암로20길"
        if detail_area_or == 1000839:
            detail_area_ko="호암로24길"
        if detail_area_or == 1000456:
            detail_area_ko="홍연길"
        if detail_area_or == 1000457:
            detail_area_ko="홍제내길"
        if detail_area_or == 1000603:
            detail_area_ko="화곡로13길"
        if detail_area_or == 1000604:
            detail_area_ko="화곡로20길"
        if detail_area_or == 1000605:
            detail_area_ko="화곡로31길"
        if detail_area_or == 1000550:
            detail_area_ko="화곡로3길"
        if detail_area_or == 1000606:
            detail_area_ko="화곡로44나길"
        if detail_area_or == 1000551:
            detail_area_ko="화곡로4길"
        if detail_area_or == 1000607:
            detail_area_ko="화곡로59길"
        if detail_area_or == 1000608:
            detail_area_ko="화곡로64길"
        if detail_area_or == 1000309:
            detail_area_ko="화랑로13길"
        if detail_area_or == 1000310:
            detail_area_ko="화랑로14길"
        if detail_area_or == 1000311:
            detail_area_ko="화랑로19길"
        if detail_area_or == 1000312:
            detail_area_ko="화랑로1길"
        if detail_area_or == 1000313:
            detail_area_ko="화랑로32길"
        if detail_area_or == 1000314:
            detail_area_ko="화랑로40길"
        if detail_area_or == 1000085:
            detail_area_ko="회나무로13길"
        if detail_area_or == 1000880:
            detail_area_ko="효령료23길"
        if detail_area_or == 1000881:
            detail_area_ko="효령로31길"
        if detail_area_or == 1000882:
            detail_area_ko="효령로34길"
        if detail_area_or == 1000883:
            detail_area_ko="효령로72길"
        if detail_area_or == 1000086:
            detail_area_ko="효창원로39길"
        if detail_area_or == 1000506:
            detail_area_ko="효창원로93길"
        if detail_area_or == 1000087:
            detail_area_ko="후암로28길"
        if detail_area_or == 1000218:
            detail_area_ko="휘경로14길"
        if detail_area_or == 1000219:
            detail_area_ko="휘경로23길"
        if detail_area_or == 1000220:
            detail_area_ko="휘경로2길"
        if detail_area_or == 1000221:
            detail_area_ko="휘경로3길"
        if detail_area_or == 1000778:
            detail_area_ko="흑설로13길"
        if detail_area_or == 1000779:
            detail_area_ko="흑설로9길"
        if detail_area_or == 1001241:
            detail_area_ko="혜화역 대학로_2"
        if detail_area_or == 1001242:
            detail_area_ko="혜화역 대학로_3"
        if detail_area_or == 1001182:
            detail_area_ko="홍익대학교 주변"
        if detail_area_or == 1001420: 
            detail_area_ko="고척근린시장"
        if detail_area_or == 1001490:
            detail_area_ko="길동복조리시장"   
        detail_area = (detail_area_or-1000001)/(1001496-1000001)
   
        #업종
        if sector_or == "CS200019":
            sector_ko="PC방"
        if sector_or == "CS300031":
            sector_ko="가구"
        if sector_or == "CS300015":
            sector_ko="가방"
        if sector_or == "CS300032":
            sector_ko="가전제품"
        if sector_or == "CS200032":
            sector_ko="가전제품수리"
        if sector_or == "CS200036":
            sector_ko="고시원"
        if sector_or == "CS200017":
            sector_ko="골프연습장"
        if sector_or == "CS200029":
            sector_ko="네일숍"
        if sector_or == "CS200037":
            sector_ko="노래방"
        if sector_or == "CS200016":
            sector_ko="당구장"
        if sector_or == "CS300021":
            sector_ko="문구"
        if sector_or == "CS300006":
            sector_ko="미곡판매"
        if sector_or == "CS200028":
            sector_ko="미용실"
        if sector_or == "CS300010":
            sector_ko="반찬가게"
        if sector_or == "CS200033":
            sector_ko="부동산중개업"
        if sector_or == "CS100008":
            sector_ko="분식전문점"
        if sector_or == "CS300020":
            sector_ko="서적"
        if sector_or == "CS300027":
            sector_ko="섬유제품"
        if sector_or == "CS200031":
            sector_ko="세탁소"
        if sector_or == "CS300008":
            sector_ko="수산물판매"
        if sector_or == "CS300001":
            sector_ko="슈퍼마켓"
        if sector_or == "CS200005":
            sector_ko="스포츠 강습"
        if sector_or == "CS200024":
            sector_ko="스포츠클럽"
        if sector_or == "CS300017":
            sector_ko="시계및귀금속"
        if sector_or == "CS300014":
            sector_ko="신발"
        if sector_or == "CS300016":
            sector_ko="안경"
        if sector_or == "CS300029":
            sector_ko="애완동물"
        if sector_or == "CS100004":
            sector_ko="양식음식점"
        if sector_or == "CS200034":
            sector_ko="여관"
        if sector_or == "CS200003":
            sector_ko="예술학원"
        if sector_or == "CS300026":
            sector_ko="완구"
        if sector_or == "CS200002":
            sector_ko="외국어학원"
        if sector_or == "CS300024":
            sector_ko="운동/경기용품"
        if sector_or == "CS300007":
            sector_ko="육류판매"
        if sector_or == "CS300019":
            sector_ko="의료기기"
        if sector_or == "CS300018":
            sector_ko="의약품"
        if sector_or == "CS300035":
            sector_ko="인테리어"
        if sector_or == "CS200001":
            sector_ko="일반교습학원"
        if sector_or == "CS300011":
            sector_ko="일반의류"
        if sector_or == "CS200006":
            sector_ko="일반의원"
        if sector_or == "CS100003":
            sector_ko="일식음식점"
        if sector_or == "CS200026":
            sector_ko="자동차미용"
        if sector_or == "CS200025":
            sector_ko="자동차수리"
        if sector_or == "CS300025":
            sector_ko="자전거 및 기타운송장비"
        if sector_or == "CS300043":
            sector_ko="전자상거래업"
        if sector_or == "CS100005":
            sector_ko="제과점"
        if sector_or == "CS300036":
            sector_ko="조명용품"
        if sector_or == "CS100002":
            sector_ko="중국음식점"
        if sector_or == "CS300033":
            sector_ko="철물점"
        if sector_or == "CS300009":
            sector_ko="청과상"
        if sector_or == "CS200007":
            sector_ko="치과의원"
        if sector_or == "CS100007":
            sector_ko="치킨전문점"
        if sector_or == "CS100010":
            sector_ko="커피-음료"
        if sector_or == "CS300003":
            sector_ko="컴퓨터및주변장치판매"
        if sector_or == "CS100006":
            sector_ko="패스트푸드점"
        if sector_or == "CS300002":
            sector_ko="편의점"
        if sector_or == "CS200030":
            sector_ko="피부관리실"
        if sector_or == "CS100001":
            sector_ko="한식음식점"
        if sector_or == "CS200008":
            sector_ko="한의원"
        if sector_or == "CS300004":
            sector_ko="핸드폰"
        if sector_or == "CS100009":
            sector_ko="호프-간이주점"
        if sector_or == "CS300022":
            sector_ko="화장품"
        if sector_or == "CS300028":
            sector_ko="화초"
        




        sector = sector_or.strip("CS") #서비스 코드 앞 CS제거
        sector =pd.to_numeric(sector) #str -> int

        if sector>=100001:
            if sector<=100010:
                sector=sector-100000

        if sector>=200001:
            if sector<=300000:
                sector=sector-200000+10

        if sector>=300001:
            if sector<=400000:
                sector=sector-300000+55

        sector = (sector-1)/(98-1)



        datas = torch.tensor([[quarter, area, detail_area, sector]], dtype=torch.float32)
   
        outputs = model(datas[0:4])

        np_outputs = outputs.cpu().detach().numpy()

        outputs_weekly = int(np_outputs.T[0]*10000000) #예측한 주중 매출    
        outputs_weekeed = int(np_outputs.T[1]*10000000) #예측한 주말 매출
        outputs_weekly = format(outputs_weekly,',')
        outputs_weekeed = format(outputs_weekeed,',')

        return render_template('result.html', q_or=quarter_or, a_or=area_or, d_a_or=detail_area_or, s_or=sector_or, a_ko=area_ko, d_a_ko=detail_area_ko, s_ko=sector_ko,  weekly_price=outputs_weekly , weekend_price = outputs_weekeed)
       

        


@app.route('/detail/', methods=['GET', 'POST'])
def detail():
    if request.method == 'GET':
        return render_template('detail.html')

    if request.method == 'POST':
        quarter_or = int(request.form['quarter'])
        area = (request.form['area'])
        detail_area = int(request.form['detail_area'])
        sector =(request.form['sector'])

        #분기
        quarter = (quarter_or-1)/(4-1)

        #상권
        if area=='A':
            area=1
            area_ko="골목상권"
        elif area=='D':
            area=2
            area_ko="발달상권"
        elif area=='R':
            area=3
            area_ko="전통시장"
        elif area=='U':
            area=4
            area_ko="관광특구"

        area=(area-1)/(4-1)
        
        #위치
        if detail_area == 1000507:
            detail_area_ko="가로공원로58길"
        if detail_area == 1000552:
            detail_area_ko="가로공원로80길"
        if detail_area == 1001419:
            detail_area_ko="가리봉시장"
        if detail_area == 1000683:
            detail_area_ko="가마산로61길"
        if detail_area == 1000566:
            detail_area_ko="강서로8길"

        detail_area = (detail_area-1000001)/(1001496-1000001)
   
        #업종
        if sector == "CS200019":
            sector_ko="PC방"
        if sector == "CS300031":
            sector_ko="가구"
        if sector == "CS300015":
            sector_ko="가방"
        if sector == "CS300032":
            sector_ko="가전제품"
        if sector == "CS200032":
            sector_ko="가전제품수리"
        if sector == "CS200036":
            sector_ko="고시원"
        if sector == "CS200002":
            sector_ko="외국어학원"



        sector = sector.strip("CS") #서비스 코드 앞 CS제거
        sector =pd.to_numeric(sector) #str -> int

        if sector>=100001:
            if sector<=100010:
                sector=sector-100000

        if sector>=200001:
            if sector<=300000:
                sector=sector-200000+10

        if sector>=300001:
            if sector<=400000:
                sector=sector-300000+55

        sector = (sector-1)/(98-1)



        datas = torch.tensor([[quarter, area, detail_area, sector]], dtype=torch.float32)
   
        outputs = model(datas[0:4])

        np_outputs = outputs.cpu().detach().numpy()


        #outputs_weekly = np_outputs.T[0]*10000000 #예측한 주중 매출
        #outputs_weekeed = np_outputs.T[1]*10000000 #예측한 주말 매출
        #outputs_weekly = np.around(np_outputs.T[0]*10000000) #예측한 주중 매출
        #outputs_weekeed = np.around(np_outputs.T[1]*10000000) #예측한 주말 매출
        outputs_weekly = int(np_outputs.T[0]*10000000) #예측한 주중 매출

        outputs_weekeed = int(np_outputs.T[1]*10000000) #예측한 주말 매출
        outputs_weekly = format(outputs_weekly,',')
        outputs_weekeed = format(outputs_weekeed,',')

        return render_template('result.html', q=quarter_or, a=area_ko, d=detail_area_ko, s=sector_ko,  weekly_price=outputs_weekly , weekend_price = outputs_weekeed)
       

        


@app.route("/ajaxfile",methods=["POST","GET"])
def ajaxfile():
    try:
        conn = mysql.connect()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        if request.method == 'POST':
            draw = request.form['draw'] 
            row = int(request.form['start'])
            rowperpage = int(request.form['length'])
            searchValue = request.form["search[value]"]
            print(draw)
            print(row)
            print(rowperpage)
            print(searchValue)
 
            ## Total number of records without filtering
            cursor.execute("select count(*) as allcount from viewinside")
            rsallcount = cursor.fetchone()
            totalRecords = rsallcount['allcount']
            print(totalRecords) 
 
            ## Total number of records with filtering
            likeString = "%" + searchValue +"%"
            cursor.execute("SELECT count(*) as allcount from viewinside WHERE title LIKE %s OR locate LIKE %s OR keyword1 LIKE %s OR keyword2 LIKE %s OR keyword3 LIKE %s OR phone LIKE %s  OR link LIKE %s", (likeString, likeString,likeString,likeString, likeString,likeString,likeString))
            rsallcount = cursor.fetchone()
            totalRecordwithFilter = rsallcount['allcount']
            print(totalRecordwithFilter) 
 
            ## Fetch records
            if searchValue=='':
                cursor.execute("SELECT * FROM viewinside ORDER BY title asc limit %s, %s;", (row, rowperpage))
                employeelist = cursor.fetchall()
            else:        
                cursor.execute("SELECT * FROM viewinside WHERE title LIKE %s OR locate LIKE %s OR keyword1 LIKE %s OR keyword2 LIKE %s OR keyword3 LIKE %s OR phone LIKE %s OR link LIKE %s limit %s, %s;", (likeString, likeString,likeString,likeString, likeString,likeString,likeString, row, rowperpage))
                employeelist = cursor.fetchall()
 
            data = []
            for row in employeelist:
                data.append({
                    'title': row['title'],
                    'locate': row['locate'],
                    'keyword1': row['keyword1'],
                    'keyword2': row['keyword2'],
                    'keyword3': row['keyword3'],
                    'phone': row['phone'],
                    'link': row['link'],
                    
                })
 
            response = {
                'draw': draw,
                'iTotalRecords': totalRecords,
                'iTotalDisplayRecords': totalRecordwithFilter,
                'aaData': data,
            }
            return jsonify(response)
    except Exception as e:
        print(e)
    finally:
        cursor.close() 
        conn.close()


@app.route("/fetchdeta",methods=["POST","GET"])
def fetchdeta():
    conn = mysql.connect()
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    if request.method == 'POST':
        search = request.form['search']
        print(search)
        query = "SELECT * from viewinside WHERE locate LIKE '{}%' LIMIT 10".format(search)
        cursor.execute(query)
        postslist = cursor.fetchall() 
        cursor.close()
    return jsonify({'htmlresponse': render_template('response.html',postslist=postslist)})

 
if __name__ == "__main__":
    app.run(debug=True)