#lang racket
(define (sort_array arr)
  (define (ones x) (for/sum ([c (in-string (number->string (abs x) 2))] #:when (char=? c #\1)) 1))
  (sort arr (lambda (a b)
              (cond [(< (ones a) (ones b)) #t]
                    [(> (ones a) (ones b)) #f]
                    [else (< a b)]))))
