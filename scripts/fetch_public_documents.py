"""Download publicly disclosed Chinese procurement contracts and tender documents.

The files stay in data/raw/ and are ignored by Git. Source URLs are official
government procurement or public-resource trading platforms.
"""

from __future__ import annotations

import concurrent.futures
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
)


@dataclass(frozen=True)
class Source:
    doc_id: str
    filename: str
    title: str
    document_type: str
    url: str


SOURCES = [
    Source(
        "DOC001",
        "doc01_beijing_yufa_monitoring_contract.pdf",
        "安装环境整治监控设备（试点）项目政府采购货物买卖合同",
        "合同",
        "http://www.ccgp-beijing.gov.cn/content_file/xxgg/qjxxgg/qjhtgg/2025/8/"
        "e647b927f8814e45a7c72e4977b8179a.pdf",
    ),
    Source(
        "DOC002",
        "doc02_ningguo_garbage_transfer_contract.pdf",
        "安徽省宁国市港口镇垃圾中转站建设工程设备采购项目（二次）第二包合同",
        "合同",
        "https://ggzyjy.xuancheng.gov.cn/xcspfront/uploadfile/"
        "9b5c3623-93fa-4285-a1d2-c3f196a96954/"
        "%E5%AE%89%E5%BE%BD%E7%9C%81%E5%AE%81%E5%9B%BD%E5%B8%82%E6%B8%AF"
        "%E5%8F%A3%E9%95%87%E5%9E%83%E5%9C%BE%E4%B8%AD%E8%BD%AC%E7%AB%99"
        "%E5%BB%BA%E8%AE%BE%E5%B7%A5%E7%A8%8B%E8%AE%BE%E5%A4%87%E9%87%87"
        "%E8%B4%AD%E9%A1%B9%E7%9B%AE%E4%BA%8C%E6%AC%A1%E7%AC%AC%E4%BA%8C"
        "%E5%8C%85.pdf",
    ),
    Source(
        "DOC003",
        "doc03_henan_geology_bigdata_tender.pdf",
        "河南省地质局河南省地质大数据平台（2025）项目招标文件",
        "招标文件",
        "https://zfcg.henan.gov.cn/cmsweb81e27e/nas/webfile2024/henan/"
        "rootfiles/2025/07/23/ebbc0e68d67d40eeb8a695b56ee59629.pdf",
    ),
    Source(
        "DOC004",
        "doc04_beijing_security_testing_contract.pdf",
        "业务系统安全及密码测评政府采购合同",
        "合同",
        "http://cgj.beijing.gov.cn/xxgk/zfcg/htgs/202505/"
        "P020250522415415879758.pdf",
    ),
    Source(
        "DOC005",
        "doc05_henan_animal_vaccine_tender.pdf",
        "河南省农业农村厅2025年动物疫病强制免疫疫苗项目招标文件",
        "招标文件",
        "https://jiyuan.zfcg.henan.gov.cn/cmsweb81e27e/henan/rootfiles/2025/04/30/"
        "df951dfb693941c6bfefd6a8e444dfda.pdf",
    ),
    Source(
        "DOC006",
        "doc06_beijing_huangcun_cleaning_contract.pdf",
        "黄村镇道路环境保洁外包服务（06包）政府采购合同",
        "合同",
        "http://www.ccgp-beijing.gov.cn/content_file/xxgg/qjxxgg/qjhtgg/2025/7/"
        "9c104db361d5411c997387321b34248e.pdf",
    ),
    Source(
        "DOC007",
        "doc07_beijing_vocational_server_contract.pdf",
        "北京电子科技职业学院2025年数据中心服务器更新购置合同",
        "合同",
        "http://www.ccgp-beijing.gov.cn/content_file/xxgg/sjxxgg/htgg/2025/8/"
        "254a90371f4f4371ab272027d3faf51b.pdf",
    ),
    Source(
        "DOC008",
        "doc08_xinyang_food_testing_tender.pdf",
        "信阳市市场监督管理局2025年市本级食品安全抽检项目招标文件",
        "招标文件",
        "https://xinyang.zfcg.henan.gov.cn/cmsweb81e27e/nas/webfile2024/"
        "xinyang/rootfiles/2025/04/09/12c81693c0d2443da38a140635c5aca2.pdf",
    ),
    Source(
        "DOC009",
        "doc09_huangchuan_grain_tender.pdf",
        "潢川县2025年粮油规模种植主体单产提升项目（一包）招标文件",
        "招标文件",
        "https://xinmi.zfcg.henan.gov.cn/cmsweb81e27e/nas/webfile2024/"
        "huangchuan/rootfiles/2025/08/08/1fa4227875794c779b7d68add51665cd.pdf",
    ),
    Source(
        "DOC010",
        "doc10_beijing_mentougou_geohazard_supervision_contract.pdf",
        "2025年门头沟区普通公路沿线地质灾害防治工程施工监理合同协议书",
        "合同",
        "https://ggzyfw.beijing.gov.cn/cmsbj/u/cms/cn.gov.bjggzyfw.www/"
        "202508/8683705230631.pdf",
    ),
    Source(
        "DOC011",
        "doc11_henan_transport_it_ops_tender.pdf",
        "河南省交通运输调度指挥中心信息化系统及基础设施运维服务项目招标文件",
        "招标文件",
        "https://zfcg.henan.gov.cn/cmsweb81e27e/nas/webfile2024/henan/"
        "rootfiles/2025/12/19/c0817b920a514f96943138ca1c6ed8cb.pdf",
    ),
    Source(
        "DOC012",
        "doc12_beijing_pinggu_highway_supervision_contract.pdf",
        "2024年平谷区普通公路沿线地质灾害防治工程施工监理合同协议书",
        "合同",
        "https://ggzyfw.beijing.gov.cn/cmsbj/u/cms/cn.gov.bjggzyfw.www/"
        "202406/2451304553700.pdf",
    ),
    Source(
        "DOC013",
        "doc13_beijing_smart_city_evaluation_contract.pdf",
        "2024年度智慧城市规划后评估服务政府采购合同",
        "合同",
        "http://www.ccgp-beijing.gov.cn/wwprotect/P0202411/P020241104/"
        "P020241104638420385593.pdf",
    ),
    Source(
        "DOC014",
        "doc14_beijing_huairou_highway_repair_contract.pdf",
        "2025年怀柔区承塔线、京漠线及宝山寺东桥中修工程合同协议书",
        "合同",
        "https://ggzyfw.beijing.gov.cn/cmsbj/u/cms/cn.gov.bjggzyfw.www/"
        "202510/5302555288747.pdf",
    ),
    Source(
        "DOC015",
        "doc15_customs_goods_contract.pdf",
        "海关总署2025年度信息化建设国产化终端设备批量采购合同",
        "合同",
        "http://shantou.customs.gov.cn/hhht_customs/fileDir/resource/cms/"
        "article/4697236/6738734/2025091617560620772.pdf",
    ),
]


def download(source: Source) -> tuple[str, str]:
    destination = RAW_DIR / source.filename
    if destination.exists() and destination.read_bytes()[:5] == b"%PDF-":
        return source.doc_id, f"已存在 {destination.name}"

    parsed = urlparse(source.url)
    request = urllib.request.Request(
        source.url,
        headers={
            "User-Agent": USER_AGENT,
            "Referer": f"{parsed.scheme}://{parsed.netloc}/",
            "Accept": "application/pdf,*/*",
        },
    )
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        if not data.startswith(b"%PDF-"):
            raise ValueError("响应不是 PDF")
        partial.write_bytes(data)
        partial.replace(destination)
        return source.doc_id, f"成功 {len(data) / 1024:.0f} KB"
    finally:
        partial.unlink(missing_ok=True)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download, source): source for source in SOURCES}
        for future in concurrent.futures.as_completed(futures):
            source = futures[future]
            try:
                doc_id, message = future.result()
                print(f"{doc_id}  {message}  {source.filename}")
            except Exception as exc:  # noqa: BLE001 - report each source independently
                failures += 1
                print(f"{source.doc_id}  失败  {type(exc).__name__}: {exc}", file=sys.stderr)
    if failures:
        raise SystemExit(f"{failures} 份下载失败")
    print(f"完成：{len(SOURCES)} 份公开文件已保存至 {RAW_DIR}")


if __name__ == "__main__":
    main()
