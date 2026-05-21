"use client";

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CircleCheck, Link2, Star, Tag, X } from "lucide-react";
import { api } from "@/lib/api";
import type { AgentRecord, CheckpointRecord } from "@/lib/api-types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { EmptyState } from "@/components/ui/empty";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { Table, TBody, TD, TH, THead, TR } from "@/components/ui/table";
import { formatNumber, formatRelativeTime } from "@/lib/format";

export function CheckpointList({
  checkpoints,
  showAttach = true,
  emptyHint,
}: {
  checkpoints: CheckpointRecord[];
  showAttach?: boolean;
  emptyHint?: string;
}) {
  if (!checkpoints.length) {
    return (
      <EmptyState
        title="No checkpoints"
        description={emptyHint ?? "Train an agent to produce a checkpoint."}
      />
    );
  }
  return (
    <Card>
      <Table>
        <THead>
          <TR>
            <TH>Step</TH>
            <TH>Tag</TH>
            <TH>Metric</TH>
            <TH>Created</TH>
            <TH className="text-right">Actions</TH>
          </TR>
        </THead>
        <TBody>
          {checkpoints.map((cp) => (
            <CheckpointRow key={cp.id} checkpoint={cp} showAttach={showAttach} />
          ))}
        </TBody>
      </Table>
    </Card>
  );
}

function CheckpointRow({
  checkpoint,
  showAttach,
}: {
  checkpoint: CheckpointRecord;
  showAttach: boolean;
}) {
  const qc = useQueryClient();
  const [editingTag, setEditingTag] = useState(false);
  const [tagDraft, setTagDraft] = useState(checkpoint.tag ?? "");
  const [attachOpen, setAttachOpen] = useState(false);

  const setTag = useMutation({
    mutationFn: (tag: string | null) => api.tagCheckpoint(checkpoint.id, tag),
    onSuccess: () => {
      setEditingTag(false);
      qc.invalidateQueries({ queryKey: ["checkpoints"] });
      qc.invalidateQueries({ queryKey: ["run-checkpoints"] });
    },
  });

  return (
    <TR>
      <TD className="tabular-nums font-mono text-xs">{formatNumber(checkpoint.step, 0)}</TD>
      <TD>
        {editingTag ? (
          <div className="flex items-center gap-1">
            <Input
              value={tagDraft}
              onChange={(e) => setTagDraft(e.target.value)}
              placeholder="tag name"
              className="h-7 w-32 text-xs"
              autoFocus
            />
            <Button
              size="icon"
              variant="ghost"
              onClick={() => setTag.mutate(tagDraft.trim() || null)}
              disabled={setTag.isPending}
              aria-label="Save tag"
            >
              <CircleCheck className="h-3.5 w-3.5 text-success" />
            </Button>
            <Button
              size="icon"
              variant="ghost"
              onClick={() => setEditingTag(false)}
              aria-label="Cancel"
            >
              <X className="h-3.5 w-3.5" />
            </Button>
          </div>
        ) : checkpoint.tag ? (
          <Badge variant={checkpoint.tag === "best" ? "success" : "primary"} className="font-mono">
            {checkpoint.tag === "best" && <Star className="h-3 w-3" />}
            {checkpoint.tag}
          </Badge>
        ) : (
          <span className="text-xs text-muted-foreground">—</span>
        )}
      </TD>
      <TD className="text-xs text-muted-foreground">
        {checkpoint.metric_name ? (
          <>
            <span className="font-mono">{checkpoint.metric_name}</span>{" "}
            <span className="tabular-nums text-foreground">
              {formatNumber(checkpoint.metric_value)}
            </span>
          </>
        ) : (
          "—"
        )}
      </TD>
      <TD className="text-xs text-muted-foreground">{formatRelativeTime(checkpoint.created_at)}</TD>
      <TD className="text-right">
        <div className="inline-flex items-center gap-1">
          {!editingTag && (
            <Button
              size="icon"
              variant="ghost"
              onClick={() => {
                setTagDraft(checkpoint.tag ?? "");
                setEditingTag(true);
              }}
              aria-label="Edit tag"
            >
              <Tag className="h-3.5 w-3.5" />
            </Button>
          )}
          {showAttach && (
            <Button
              size="sm"
              variant="secondary"
              onClick={() => setAttachOpen((v) => !v)}
              className="text-xs"
            >
              <Link2 className="h-3.5 w-3.5" />
              Attach
            </Button>
          )}
        </div>
        {attachOpen && (
          <div className="mt-2">
            <AttachPicker
              checkpoint={checkpoint}
              onDone={() => setAttachOpen(false)}
            />
          </div>
        )}
      </TD>
    </TR>
  );
}

function AttachPicker({
  checkpoint,
  onDone,
}: {
  checkpoint: CheckpointRecord;
  onDone: () => void;
}) {
  const qc = useQueryClient();
  const { data: agents } = useQuery({ queryKey: ["agents"], queryFn: api.listAgents });
  const [agentId, setAgentId] = useState<string>("");
  const attach = useMutation({
    mutationFn: (id: string) => api.attachCheckpoint(checkpoint.id, id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["agents"] });
      qc.invalidateQueries({ queryKey: ["agent"] });
      onDone();
    },
  });

  const eligible = agents ?? [];

  if (eligible.length === 0) {
    return (
      <p className="text-xs text-muted-foreground">
        No agents yet — create one first.
      </p>
    );
  }

  return (
    <div className="flex items-center justify-end gap-2 text-left">
      <Select
        value={agentId}
        onChange={(e) => setAgentId(e.target.value)}
        className="h-8 w-56 text-xs"
      >
        <option value="">— select agent —</option>
        {eligible.map((a: AgentRecord) => (
          <option key={a.id} value={a.id}>
            {a.name} ({a.config.agent_type}/{a.config.network_preset})
          </option>
        ))}
      </Select>
      <Button
        size="sm"
        disabled={!agentId || attach.isPending}
        onClick={() => attach.mutate(agentId)}
      >
        {attach.isPending ? "Attaching…" : "Confirm"}
      </Button>
      <Button size="sm" variant="ghost" onClick={onDone}>
        Cancel
      </Button>
    </div>
  );
}
