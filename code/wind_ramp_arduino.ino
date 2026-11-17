/*
 * wind_ramp_arduino.ino —— 风电爬坡预警系统 Arduino 采集端(备用存档, 暂未集成)
 * ============================================================
 * 作用: 采集风速/风向/温度/功率, 按文本帧协议经串口上传给 PC 端 serial_parser.py。
 *
 * 硬件: Arduino Uno + 风速传感器 + 风向传感器 + (可选)温度传感器 + 功率模拟/读取
 * 协议: 见 PC 端 serial_parser.py 头部注释; 每帧一行, 逗号分隔, *XX 为 XOR 校验。
 *
 * 引脚分配(按实际接线修改):
 *   A0 风速传感器(模拟量, 一般 0-5V 对应 0-30m/s)
 *   A1 风向传感器(模拟量, 0-5V 对应 0-360deg)
 *   A2 环境温度(可选, 如 LM35: 10mV/deg)
 *   A3 机舱温度(可选)
 *   D2 风速脉冲输入(若用风杯式脉冲传感器; 本示例用模拟量, 此处保留)
 *
 * 说明: Pab1/2/3(叶片角)、Prtv(变桨)、C_mw/P_mw(容量/功率) 若无对应传感,
 *       可固定为常量或由 PC 端覆盖; 本示例给出占位值。
 */

const int PIN_WSPD = A0;
const int PIN_WDIR = A1;
const int PIN_ETMP = A2;
const int PIN_ITMP = A3;

// 采样周期: 10min = 600000ms(演示可改小, 如 2000ms 加速)
const unsigned long SAMPLE_INTERVAL_MS = 600000UL;
unsigned long lastSample = 0;

// 风速标定: ADC(0-1023) -> m/s, 假设 0-5V 对应 0-30 m/s
float adcToWspd(int adc) {
  float v = adc * (5.0 / 1023.0);
  return v * (30.0 / 5.0);
}

// 风向: ADC -> 0-360 deg
float adcToWdir(int adc) {
  float v = adc * (5.0 / 1023.0);
  return v * (360.0 / 5.0);
}

// 温度(LM35): 10mV/deg -> deg C
float adcToTemp(int adc) {
  return adc * (5.0 / 1023.0) * 100.0;
}

// XOR 校验: 对 $ 与 * 之间字符逐字节异或
byte xorChecksum(const char *s) {
  byte c = 0;
  while (*s) { c ^= (byte)(*s); s++; }
  return c;
}

void sendFrame(float wspd, float wdir, float etmp, float itmp,
               float ndir, float pab1, float pab2, float pab3,
               float prtv, float cmw, float pmw) {
  // 拼 body(不含 $ 的校验起点)与完整帧
  char payload[160];
  // 注意: 协议里 chk 对 body[1:](去掉 $)异或; 这里先拼去掉 $ 的部分算校验
  snprintf(payload, sizeof(payload),
           "W,%0.3f,%0.2f,%0.3f,%0.3f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.3f,%0.3f",
           wspd, wdir, etmp, itmp, ndir, pab1, pab2, pab3, prtv, cmw, pmw);
  byte chk = xorChecksum(payload);
  Serial.print("$");
  Serial.print(payload);
  Serial.print("*");
  if (chk < 16) Serial.print("0");
  Serial.println(chk, HEX);
}

void setup() {
  Serial.begin(115200);
  while (!Serial) { ; }
  Serial.println("# wind_ramp_arduino ready");
}

void loop() {
  unsigned long now = millis();
  if (now - lastSample >= SAMPLE_INTERVAL_MS || lastSample == 0) {
    lastSample = now;

    int a0 = analogRead(PIN_WSPD);
    int a1 = analogRead(PIN_WDIR);
    int a2 = analogRead(PIN_ETMP);
    int a3 = analogRead(PIN_ITMP);

    float wspd = adcToWspd(a0);
    float wdir = adcToWdir(a1);
    float etmp = adcToTemp(a2);
    float itmp = adcToTemp(a3);

    // 占位: 机舱方向/叶片角/变桨/容量/功率(按实际替换)
    float ndir = wdir;
    float pab1 = 10.0, pab2 = 10.0, pab3 = 10.0;
    float prtv = 0.0;
    float cmw = 201.0;
    // 简化功率模型(仅演示): P ~ 0.5 * rho * A * v^3, 归一后占位
    float pmw = min(cmw, max(0.0, 0.05 * wspd * wspd * wspd));

    sendFrame(wspd, wdir, etmp, itmp, ndir, pab1, pab2, pab3, prtv, cmw, pmw);
  }
}
