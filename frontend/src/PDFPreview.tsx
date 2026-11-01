import { useEffect, useRef, useState } from "react";
import * as pdfjs from "pdfjs-dist";
import pdfWorker from "pdfjs-dist/build/pdf.worker.min.mjs?url";
import type { Citation } from "./types";
pdfjs.GlobalWorkerOptions.workerSrc = pdfWorker;

export default function PDFPreview({
  url,
  citation,
}: {
  url: string;
  citation: Citation;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [error, setError] = useState("");
  const locator = citation.locator.find((l) => l.page) ?? {};
  useEffect(() => {
    let stopped = false;
    const task = pdfjs.getDocument(url);
    let render: ReturnType<pdfjs.PDFPageProxy["render"]> | undefined;
    task.promise
      .then(async (doc) => {
        const page = await doc.getPage(Number(locator.page ?? 1));
        if (stopped || !canvas.current) return;
        const viewport = page.getViewport({
          scale: Math.min(1.4, 580 / page.getViewport({ scale: 1 }).width),
        });
        const el = canvas.current;
        el.width = viewport.width;
        el.height = viewport.height;
        const ctx = el.getContext("2d")!;
        render = page.render({ canvas: el, canvasContext: ctx, viewport });
        await render.promise;
        if (stopped) return;
        for (const loc of citation.locator) {
          if (loc.page !== locator.page || !Array.isArray(loc.bbox)) continue;
          const [x1, y1, x2, y2] = loc.bbox as number[];
          ctx.fillStyle = "rgba(207,164,68,.22)";
          ctx.strokeStyle = "#a97825";
          ctx.fillRect(
            (x1 / 1000) * el.width,
            (y1 / 1000) * el.height,
            ((x2 - x1) / 1000) * el.width,
            ((y2 - y1) / 1000) * el.height,
          );
          ctx.strokeRect(
            (x1 / 1000) * el.width,
            (y1 / 1000) * el.height,
            ((x2 - x1) / 1000) * el.width,
            ((y2 - y1) / 1000) * el.height,
          );
        }
      })
      .catch((e) => {
        if (!stopped) setError(String(e.message));
      });
    return () => {
      stopped = true;
      render?.cancel();
      void task.destroy();
    };
  }, [url, citation]);
  return (
    <div className="pdf-preview">
      <small>
        第 {String(locator.page ?? 1)} 页
        {locator.bbox ? " · 引用区域已标注" : " · 当前解析结果无区域坐标"}
      </small>
      {error ? <p role="alert">{error}</p> : <canvas ref={canvas} />}
    </div>
  );
}
