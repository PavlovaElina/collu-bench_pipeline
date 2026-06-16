#lang racket
(define (sort_array arr)
  (cond [(null? arr) arr]
        [else (define s (+ (first arr) (last arr)))
              (if (odd? s) (sort arr <) (sort arr >))]))
