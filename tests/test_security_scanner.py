"""Tests for security scanner."""

from crusades.security.scanner import CodeScanner


class TestCodeScanner:
    def setup_method(self):
        self.scanner = CodeScanner()

    def test_valid_code(self):
        code = '''
import torch
import torch.nn.functional as F

def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    total_tokens = 0
    final_logits = None
    final_loss = 0.0
    for step in range(num_steps):
        batch = next(data_iterator).to(device)
        input_ids = batch[:, :-1]
        labels = batch[:, 1:]
        outputs = model(input_ids)
        logits = outputs.logits
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        loss.backward()
        if optimizer:
            optimizer.step()
            optimizer.zero_grad()
        total_tokens += batch.numel()
        final_logits = logits.detach()
        final_loss = loss.item()
    from dataclasses import dataclass
    @dataclass
    class Result:
        final_logits: object
        total_tokens: int
        final_loss: float
    return Result(final_logits=final_logits, total_tokens=total_tokens, final_loss=final_loss)
'''
        result = self.scanner.validate(code)
        assert result.valid is True

    def test_forbidden_import_os(self):
        code = '''
import os
def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    pass
'''
        result = self.scanner.validate(code)
        assert result.valid is False
        assert any("forbidden import" in e for e in result.errors)

    def test_forbidden_import_subprocess(self):
        code = '''
import subprocess
def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    pass
'''
        result = self.scanner.validate(code)
        assert result.valid is False

    def test_missing_inner_steps(self):
        code = "x = 1"
        result = self.scanner.validate(code)
        assert result.valid is False
        assert any("inner_steps" in e for e in result.errors)

    def test_syntax_error(self):
        code = "def ("
        result = self.scanner.validate(code)
        assert result.valid is False
        assert any("Syntax error" in e for e in result.errors)

    def test_forbidden_exec(self):
        code = '''
def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    exec("x=1")
'''
        result = self.scanner.validate(code)
        assert result.valid is False

    def test_forbidden_builtins_access(self):
        code = '''
def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    x = __builtins__
'''
        result = self.scanner.validate(code)
        assert result.valid is False

    def test_forbidden_eval(self):
        code = '''
def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    eval("1+1")
'''
        result = self.scanner.validate(code)
        assert result.valid is False

    def test_forbidden_setattr(self):
        code = '''
def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    setattr(model, "x", 1)
'''
        result = self.scanner.validate(code)
        assert result.valid is False

    def test_inner_steps_too_few_args(self):
        code = '''
def inner_steps(model, data_iterator):
    pass
'''
        result = self.scanner.validate(code)
        assert result.valid is False
        assert any("args" in e for e in result.errors)

    def test_forbidden_torch_load(self):
        code = '''
import torch
def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    torch.load("model.pt")
'''
        result = self.scanner.validate(code)
        assert result.valid is False

    def test_forbidden_string_in_literal(self):
        code = '''
import torch
def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    x = "__builtins__"
'''
        result = self.scanner.validate(code)
        assert result.valid is False

    def test_torch_alias_forbidden(self):
        code = '''
import torch as t
def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    pass
'''
        result = self.scanner.validate(code)
        assert result.valid is False
