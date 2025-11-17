#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import http.server
import socketserver
import os

if __name__ == "__main__":
    # Sirve la carpeta que contiene este archivo (debe ejecutarse desde el repo raíz o desde /web)
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.dirname(here))  # subir al root del proyecto
    PORT = 8000
    Handler = http.server.SimpleHTTPRequestHandler
    print(f"Sirviendo en http://localhost:{PORT}/web/ (Ctrl+C para salir)")
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.server_close()


