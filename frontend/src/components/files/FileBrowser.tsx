"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ChevronUp, File, Folder, Loader2 } from "lucide-react";
import { api } from "@/lib/api";
import type { FileEntry } from "@/lib/api-types";
import { Dialog } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/cn";

export function FileBrowser({
  open,
  onOpenChange,
  ext = ".ptm",
  onSelect,
  initialPath,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  ext?: string;
  onSelect: (entry: FileEntry) => void;
  initialPath?: string;
}) {
  const [path, setPath] = useState<string | undefined>(initialPath);
  const [picked, setPicked] = useState<FileEntry | null>(null);
  const { data, isLoading, error } = useQuery({
    queryKey: ["files-browse", path, ext, open],
    queryFn: () => api.browse(path, ext),
    enabled: open,
  });

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        onOpenChange(o);
        if (!o) setPicked(null);
      }}
      title={`Pick a ${ext} file`}
      description="Navigate to a checkpoint, then select it."
      width="max-w-3xl"
      footer={
        <>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            disabled={!picked || picked.is_dir}
            onClick={() => {
              if (picked && !picked.is_dir) {
                onSelect(picked);
                onOpenChange(false);
                setPicked(null);
              }
            }}
          >
            Use this file
          </Button>
        </>
      }
    >
      <div className="flex items-center gap-2 px-1 py-2 text-xs text-muted-foreground">
        <span className="font-mono truncate">{data?.path ?? path ?? "…"}</span>
      </div>
      <div className="overflow-auto rounded-md border border-border">
        {isLoading && (
          <div className="flex items-center justify-center p-6 text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
          </div>
        )}
        {error && (
          <p className="p-4 text-sm text-destructive">
            {(error as Error).message}
          </p>
        )}
        {data && (
          <ul className="divide-y divide-border">
            {data.parent && data.parent !== data.path && (
              <li
                className="flex cursor-pointer items-center gap-2 px-3 py-2 text-sm hover:bg-muted/60"
                onClick={() => {
                  setPath(data.parent ?? undefined);
                  setPicked(null);
                }}
              >
                <ChevronUp className="h-4 w-4 text-muted-foreground" />
                <span className="text-muted-foreground">.. (up)</span>
              </li>
            )}
            {data.entries.length === 0 && (
              <li className="p-4 text-sm text-muted-foreground">
                No items match.
              </li>
            )}
            {data.entries.map((entry) => {
              const isSelected = picked?.path === entry.path;
              return (
                <li
                  key={entry.path}
                  className={cn(
                    "flex cursor-pointer items-center justify-between gap-3 px-3 py-2 text-sm hover:bg-muted/60",
                    isSelected && "bg-primary/10 text-primary"
                  )}
                  onClick={() => {
                    if (entry.is_dir) {
                      setPath(entry.path);
                      setPicked(null);
                    } else {
                      setPicked(entry);
                    }
                  }}
                  onDoubleClick={() => {
                    if (!entry.is_dir) {
                      onSelect(entry);
                      onOpenChange(false);
                      setPicked(null);
                    }
                  }}
                >
                  <div className="flex min-w-0 items-center gap-2">
                    {entry.is_dir ? (
                      <Folder className="h-4 w-4 text-muted-foreground" />
                    ) : (
                      <File className="h-4 w-4 text-muted-foreground" />
                    )}
                    <span className="truncate">{entry.name}</span>
                  </div>
                  <div className="flex shrink-0 items-center gap-3 text-xs text-muted-foreground">
                    {entry.size !== null && entry.size !== undefined && (
                      <span className="font-mono tabular-nums">{formatSize(entry.size)}</span>
                    )}
                    {entry.modified && (
                      <span className="font-mono">
                        {new Date(entry.modified).toLocaleString()}
                      </span>
                    )}
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </Dialog>
  );
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}
