// Web Serial wrapper: conexión, envío de comandos R/Z/S y lectura de telemetría Y

export class SerialManager {
  constructor() {
    this.port = null;
    this.reader = null;
    this.writer = null;
    this.textDecoder = new TextDecoderStream();
    this.textEncoder = new TextEncoder();
    this.readableClosed = null;
    this.keepReading = false;
    this.consoleCallback = null; // function(line: string)
    this.onTelemetry = null;     // function(obj)
  }

  async connect(baud) {
    if (!('serial' in navigator)) {
      throw new Error('Web Serial API no disponible en este navegador.');
    }
    this.port = await navigator.serial.requestPort();
    await this.port.open({ baudRate: baud });

    const readableStreamClosed = this.port.readable.pipeTo(this.textDecoder.writable);
    this.readableClosed = readableStreamClosed;
    this.reader = this.textDecoder.readable.getReader();

    this.writer = this.port.writable.getWriter();
    this.keepReading = true;
    this._readLoop();
  }

  async disconnect() {
    try { await this.writeLine('S'); } catch {}
    this.keepReading = false;
    try { if (this.reader) await this.reader.cancel(); } catch {}
    try { if (this.readableClosed) await this.readableClosed.catch(() => {}); } catch {}
    try { if (this.writer) this.writer.releaseLock(); } catch {}
    try { if (this.port) await this.port.close(); } catch {}
    this.port = null;
    this.reader = null;
    this.writer = null;
  }

  async writeLine(text) {
    if (!this.writer) throw new Error('No hay escritor serial.');
    const data = this.textEncoder.encode(text.endsWith('\n') ? text : (text + '\n'));
    await this.writer.write(data);
    if (this.consoleCallback) this.consoleCallback(`TX ${text.trim()}`);
  }

  async sendS() { await this.writeLine('S'); }

  async sendR(theta1, theta2, tpcSeconds) {
    // R,th1,th2,<pc_time>
    const line = `R,${theta1.toFixed(6)},${theta2.toFixed(6)},${(tpcSeconds ?? 0).toFixed(6)}`;
    await this.writeLine(line);
  }

  async sendP(kp1, kp2) {
    const line = `P,${Number(kp1).toFixed(6)},${Number(kp2).toFixed(6)}`;
    await this.writeLine(line);
  }
  async sendI(ki1, ki2) {
    const line = `I,${Number(ki1).toFixed(6)},${Number(ki2).toFixed(6)}`;
    await this.writeLine(line);
  }
  async sendD(kd1, kd2) {
    const line = `D,${Number(kd1).toFixed(6)},${Number(kd2).toFixed(6)}`;
    await this.writeLine(line);
  }

  async _readLoop() {
    let buffer = '';
    while (this.keepReading && this.reader) {
      try {
        const { value, done } = await this.reader.read();
        if (done || value === undefined) break;
        buffer += value;
        let idx;
        while ((idx = buffer.indexOf('\n')) >= 0) {
          const line = buffer.slice(0, idx).trim();
          buffer = buffer.slice(idx + 1);
          if (!line) continue;
          // Consola enfocada
          if (this.consoleCallback) this.consoleCallback(`RX ${line}`);
          // Telemetría Y
          if (line.startsWith('Y,')) {
            const parts = line.split(',');
            if (parts.length >= 8) {
              const obj = {
                arduino_ms: parseInt(parts[1], 10),
                q1: parseFloat(parts[2]),
                q2: parseFloat(parts[3]),
                q1_ref: parseFloat(parts[4]),
                q2_ref: parseFloat(parts[5]),
                u1: parseFloat(parts[6]),
                u2: parseFloat(parts[7]),
                pc_time_s: performance.now() / 1000.0
              };
              if (this.onTelemetry) this.onTelemetry(obj);
            }
          } else if (line.indexOf('theta1:') >= 0 && line.indexOf('theta2:') >= 0) {
            // Telemetría humana: "theta1: X  theta2: Y  ref1: A  ref2: B ..."
            // Soportar distintos espacios; solo extraemos lo esencial
            try {
              const rx = /theta1:\s*([-\d.]+)\s+theta2:\s*([-\d.]+)(?:\s+ref1:\s*([-\d.]+)\s+ref2:\s*([-\d.]+))?/;
              const m = line.match(rx);
              if (m) {
                const q1 = parseFloat(m[1]);
                const q2 = parseFloat(m[2]);
                const q1r = m[3] !== undefined ? parseFloat(m[3]) : NaN;
                const q2r = m[4] !== undefined ? parseFloat(m[4]) : NaN;
                const obj = {
                  arduino_ms: NaN,
                  q1, q2, q1_ref: q1r, q2_ref: q2r,
                  u1: NaN, u2: NaN,
                  pc_time_s: performance.now() / 1000.0
                };
                if (this.onTelemetry) this.onTelemetry(obj);
              }
            } catch (_) {
              // ignora si no matchea
            }
          }
        }
      } catch (e) {
        // fin del loop si se desconecta
        break;
      }
    }
  }
}


